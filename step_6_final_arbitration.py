#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_6_final_arbitration.py
@Author: WB
@Version: v2.0
@Date: 2025-8-28
@Description: 独立的最终仲裁脚本，对 is_same=False 的记录进行最终裁决。
"""

import pandas as pd
import json
import re
import os
import threading
from google import genai
from queue import Queue
from google.genai import types
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import stop_after_attempt
from tqdm import tqdm
import sys
sys.path.append('C:/Users/YYMF/PythonProjects/PI&Site/SITE_APP')
from config_utils import (
    Config, BATCH_ARBITRATE_PROMPT, shutdown_event, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential
)


def step_6_ai_arbitration(config: Config):
    """
    对 step_5 中判定为 is_same=False 的记录进行最终仲裁。
    """
    logger.info("\n" + "="*50)
    logger.info("STEP 6: 正在对不一致匹配进行最终仲裁...")
    logger.info("="*50)

    try:
        judge_results_df = pd.read_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE)
        arbitration_input_df = judge_results_df[judge_results_df['is_same'] == False].copy()

        if arbitration_input_df.empty:
            logger.info("没有需要进行最终仲裁的记录。")
            pd.DataFrame(columns=['original_id', 'decision', 'final_site', 'decision_reason']).to_parquet(config.ARBITRATION_OUTPUT_PARQUET_FILE, index=False)
            return True

        all_matches_df = pd.read_parquet(config.ALL_MATCHES_COMBINED_FILE)
        
        arbitration_input_df = pd.merge(
            arbitration_input_df,
            all_matches_df[['original_id', 'affiliation','state', 'city']],
            on='original_id',
            how='left'
        )
        logger.info(f"共找到 {len(arbitration_input_df)} 条记录需要进行仲裁。")
        
    except FileNotFoundError as e:
        logger.error(f"未找到输入文件: {e}。请先完整运行主流程的步骤1-5。")
        return False
        
    tasks, _ = get_tasks_to_process_generic(
        input_df=arbitration_input_df, 
        id_column='original_id',
        output_jsonl_path=config.ARBITRATION_OUTPUT_JSONL_FILE
    )

    @sleep_and_retry  # noqa: F821
    @limits(calls=config.CALLS_PER_MINUTE, period=config.ONE_MINUTE)
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=20))
    def query_llm_batch(tasks_batch: List[Dict]) -> Dict:
        if shutdown_event.is_set():
            return {"results": []}
        user_content_json = arbitrate_content_generator(tasks_batch)
        if not user_content_json:
            return {"results": []}
        try:
            client = genai.Client(api_key=config.GENAI_API_KEY, http_options=types.HttpOptions(base_url=config.GENAI_BASE_URL))
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config_genai = types.GenerateContentConfig(tools=[grounding_tool], system_instruction=BATCH_ARBITRATE_PROMPT, response_modalities=["TEXT"])
            response = client.models.generate_content(model=config.ARBITRATE_MODEL, contents=user_content_json, config=config_genai)
            
            if not response or not hasattr(response, 'text') or not response.text:
                raise ValueError("Gemini 返回的响应为空或无效")
            content = response.text.strip()
            cleaned_content = re.sub(r'^```json\s*|```\s*$', '', content, flags=re.MULTILINE).strip()
            logger.info(f"收到API响应 (前200字符): {cleaned_content[:200]}...")
            if not cleaned_content:
                raise ValueError("清理后响应内容为空")
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    logger.warning("直接解析JSON失败，尝试提取嵌入的JSON对象...")
                    return json.loads(json_match.group())
                raise ValueError(f"无法从响应中解析出有效的JSON内容: {cleaned_content[:200]}...")
        except Exception as e:
            logger.error(f"API 调用或解析过程中发生严重错误: {str(e)}")
            raise
        
    def arbitrate_content_generator(batch: List[Dict]) -> str:
        inputs_for_llm = []
        for task in batch:
            inputs_for_llm.append({
                "original_id": task.get("original_id"),
                "affiliation": task.get("affiliation"),
                "state": task.get("state"),
                "city": task.get("city"),
                "pi_site_name": task.get("pi_site_name"),
                "matched_site": task.get("matched_site")
            })
        return json.dumps({"inputs": inputs_for_llm}, ensure_ascii=False)
    
    def arbitrate_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
        batch_map = {task.get("original_id"): task for task in batch}
        llm_outputs = response.get('results', [])
        parsed_results = []
        processed_ids = set()

        for item in llm_outputs:
            result_id = item.get("original_id")
            if result_id in batch_map:
                parsed_results.append({
                    "status": "success", 
                    "original_id": result_id,
                    "decision": item.get("decision"),
                    "final_site": item.get("final_site"),
                    "decision_reason": item.get("decision_reason")
                })
                processed_ids.add(result_id)
            else:
                logger.warning(f"LLM在仲裁步骤返回了一个无法关联的未知ID: {result_id}")
        
        for task in batch:
            if task['original_id'] not in processed_ids:
                parsed_results.append({
                    "status": "failed", 
                    "error": "LLM response missing", 
                    "original_id": task['original_id'],
                    "decision": "neither",
                    "final_site": None,
                    "decision_reason": "LLM响应中缺少此条记录的仲裁结果。"
                })
        
        return parsed_results

    # --- 执行AI处理循环 ---
    if not tasks:
        logger.info("所有仲裁任务均已处理，无需新操作。")
    else:
        # 将并发处理逻辑封装在一个独立的循环函数中
        def run_arbitration_loop(tasks_to_run: List[Dict]):
            local_save_queue = Queue()
            task_chunks = [tasks_to_run[i:i + config.ARBITRATE_BATCH_SIZE] for i in range(0, len(tasks_to_run), config.ARBITRATE_BATCH_SIZE)]
            logger.info(f"开始处理 {len(tasks_to_run)} 个新任务，分为 {len(task_chunks)} 个批次...")
            saver_thread = threading.Thread(target=async_saver_local, args=(config.ARBITRATION_OUTPUT_JSONL_FILE, local_save_queue), daemon=True)
            saver_thread.start()
            total_processed_count = 0
            try:
                with tqdm(total=len(task_chunks), desc="AI仲裁进度") as pbar:
                    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                        future_to_batch = {executor.submit(query_llm_batch, chunk): chunk for chunk in task_chunks}
                        for future in as_completed(future_to_batch):
                            if shutdown_event.is_set():
                                for f in future_to_batch: f.cancel()
                                break
                            batch_input = future_to_batch[future]
                            try:
                                llm_response = future.result()
                                processed_results = arbitrate_result_parser(llm_response, batch_input)
                                for result in processed_results:
                                    local_save_queue.put(result)
                                successful_in_batch = sum(1 for r in processed_results if r.get('status') == 'success')
                                total_processed_count += successful_in_batch
                                pbar.set_description(f"AI仲裁进度 (批次成功: {successful_in_batch}/{len(batch_input)})")
                            except Exception as e:
                                logger.error(f"批次处理最终失败: {str(e)}")
                                for task in batch_input:
                                    local_save_queue.put({"status": "failed", "error": str(e), **task})
                            pbar.update(1)
            except KeyboardInterrupt:
                logger.warning("\n检测到中断信号 (Ctrl+C)，正在准备关闭...")
                shutdown_event.set()
            finally:
                logger.info("正在进行最后的清理和保存工作...")
                local_save_queue.join()
                local_save_queue.put(None)
                saver_thread.join(timeout=10)
                logger.info(f"--- 本次运行总计 ---")
                logger.info(f"成功处理并保存了 {total_processed_count} 条新记录。")
        
        run_arbitration_loop(tasks)
    
    # --- 收尾工作 ---
    cleanup_jsonl_file(config.ARBITRATION_OUTPUT_JSONL_FILE, id_column='original_id')
    
    try:
        if os.path.exists(config.ARBITRATION_OUTPUT_JSONL_FILE):
            final_df = pd.read_json(config.ARBITRATION_OUTPUT_JSONL_FILE, lines=True)
            if not final_df.empty:
                final_df.to_parquet(config.ARBITRATION_OUTPUT_PARQUET_FILE, index=False)
                logger.info(f"最终仲裁结果已保存到 {config.ARBITRATION_OUTPUT_PARQUET_FILE}")
            else:
                pd.DataFrame(columns=['original_id', 'decision', 'final_site', 'decision_reason']).to_parquet(config.ARBITRATION_OUTPUT_PARQUET_FILE, index=False)
                logger.info("没有成功的仲裁结果被保存，已创建空的Parquet文件。")
        else:
            pd.DataFrame(columns=['original_id', 'decision', 'final_site', 'decision_reason']).to_parquet(config.ARBITRATION_OUTPUT_PARQUET_FILE, index=False)
            logger.info("没有生成仲裁结果文件，已创建空的Parquet文件。")
    except Exception as e:
        logger.error(f"保存最终仲裁结果时发生错误: {e}")
        return False

    logger.info("STEP 6 完成。")
    return True


if __name__ == "__main__":
    config = Config()
    step_6_ai_arbitration(config)