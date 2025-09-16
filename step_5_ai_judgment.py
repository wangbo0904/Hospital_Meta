#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_5_ai_judgment.py
@Author: WB
@Version: v2.1
@Date: 2025-8-27
@Description: AI judgment of inconsistent institution names using genai, with a persistent loop to ensure all tasks are completed.
"""

import pandas as pd
import json
import re
import threading
import time  # --- MODIFICATION: 引入 time 模块 ---
from google import genai
from queue import Queue
from google.genai import types
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import stop_after_attempt
from tqdm import tqdm
from config_utils import Config, BATCH_JUDGE_PROMPT, shutdown_event, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential

def step_5_ai_judgment(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 5: 使用 AI 判断不一致的机构名...")
    logger.info("="*50)

    try:
        judge_input_df = pd.read_parquet(config.JUDGE_INPUT_FILE)
        if judge_input_df.empty:
            logger.info("没有需要AI判断的不一致记录。跳过STEP 5。")
            # --- MODIFICATION: 即使跳过，也创建一个空的输出文件 ---
            pd.DataFrame(columns=['esid', 'pi_site_name', 'matched_site', 'is_same', 'reason']).to_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE, index=False)
            return True
        if 'esid' not in judge_input_df.columns:
            logger.error(f"输入文件缺少 'esid' 列")
            return False
    except FileNotFoundError:
        logger.error(f"未找到输入文件 {config.JUDGE_INPUT_FILE}")
        return False

    # --- MODIFICATION START: 将核心处理逻辑放入 while 循环 ---
    processing_round = 1
    while not shutdown_event.is_set():
        logger.info("-" * 50)
        logger.info(f"开始第 {processing_round} 轮 AI 判断处理...")

        tasks, _ = get_tasks_to_process_generic(
            input_df=judge_input_df,
            id_column='esid',
            output_jsonl_path=config.FINAL_JUDGE_OUTPUT_JSONL_FILE,
            output_id_column='esid'
        )

        if not tasks:
            logger.info("所有 AI 判断任务均已成功处理，程序退出。")
            break

        logger.info(f"本轮发现 {len(tasks)} 个待处理任务。")

        @sleep_and_retry
        @limits(calls=config.CALLS_PER_MINUTE, period=config.ONE_MINUTE)
        @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=20))
        def query_llm_batch(tasks_batch: List[Dict]) -> Dict:
            if shutdown_event.is_set():
                return {"results": [{"status": "cancelled", **task} for task in tasks_batch]}

            user_content_json = judge_content_generator(tasks_batch)
            if not user_content_json:
                return {"results": []}

            try:
                client = genai.Client(
                    api_key=config.GENAI_API_KEY,
                    http_options=types.HttpOptions(base_url=config.GENAI_BASE_URL)
                )
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config_genai = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    system_instruction=BATCH_JUDGE_PROMPT,
                    response_modalities=["TEXT"]
                )
                response = client.models.generate_content(
                    model=config.AI_JUDGE_MODEL,
                    contents=user_content_json,
                    config=config_genai
                )
                if not response or not hasattr(response, 'text') or not response.text:
                    raise ValueError("Gemini 返回的响应为空")
                content = response.text.strip()
                cleaned_content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                cleaned_content = re.sub(r'```\s*$', '', cleaned_content, flags=re.MULTILINE).strip()
                logger.info(f"收到响应: {cleaned_content[:200]}...")
                if not cleaned_content:
                    raise ValueError("清理后的响应内容为空")
                try:
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    raise ValueError(f"无法解析JSON内容: {cleaned_content[:200]}...")
            except Exception as e:
                logger.error(f"API 调用失败: {str(e)}")
                if "404" in str(e):
                    logger.error("API 返回 404 Not Found，可能的模型名称或端点错误")
                raise

        def judge_content_generator(batch: List[Dict]) -> str:
            inputs_for_llm = []
            for task in batch:
                inputs_for_llm.append({
                    "esid": task.get("esid"),
                    "pi_site_name": task.get("pi_site_name", ""),
                    "matched_site": task.get("matched_site", "")
                })
            return json.dumps({"inputs": inputs_for_llm}, ensure_ascii=False)

        def judge_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
            batch_map = {task.get("esid"): task for task in batch}
            llm_outputs = response.get('results', [])
            parsed_results = []
            processed_ids = set()

            for item in llm_outputs:
                result_id = item.get("esid")
                if result_id in batch_map:
                    task = batch_map[result_id]
                    parsed_results.append({
                        "status": "success",
                        "esid": result_id,
                        "pi_site_name": task.get("pi_site_name"),
                        "matched_site": task.get("matched_site"),
                        "is_same": item.get("is_same"),
                        "reason": item.get("reason", "")
                    })
                    processed_ids.add(result_id)
                else:
                    logger.warning(f"LLM返回未知ID: {result_id}")

            for task in batch:
                if task['esid'] not in processed_ids:
                    parsed_results.append({
                        "status": "failed",
                        "esid": task['esid'],
                        "pi_site_name": task.get("pi_site_name"),
                        "matched_site": task.get("matched_site"),
                        "error": "LLM response missing"
                    })
            return parsed_results

        local_save_queue = Queue()
        task_chunks = [tasks[i:i + config.AI_JUDGE_BATCH_SIZE] for i in range(0, len(tasks), config.AI_JUDGE_BATCH_SIZE)]
        
        saver_thread = threading.Thread(target=async_saver_local, args=(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, local_save_queue), daemon=True)
        saver_thread.start()

        total_processed_count = 0
        try:
            with tqdm(total=len(task_chunks), desc=f"AI判断进度 (第 {processing_round} 轮)") as pbar:
                with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                    future_to_batch = {executor.submit(query_llm_batch, chunk): chunk for chunk in task_chunks}
                    for future in as_completed(future_to_batch):
                        if shutdown_event.is_set():
                            for f in future_to_batch:
                                f.cancel()
                            break
                        batch_input = future_to_batch[future]
                        try:
                            llm_response = future.result()
                            processed_results = judge_result_parser(llm_response, batch_input)
                            for result in processed_results:
                                local_save_queue.put(result)
                            successful_in_batch = sum(1 for r in processed_results if r.get('status') == 'success')
                            total_processed_count += successful_in_batch
                            pbar.set_description(f"AI判断进度 (批次成功: {successful_in_batch}/{len(batch_input)})")
                        except Exception as e:
                            logger.error(f"批次处理失败: {str(e)}")
                            for task in batch_input:
                                local_save_queue.put({"status": "failed", "error": str(e), **task})
                        pbar.update(1)
        except KeyboardInterrupt:
            logger.info("检测到中断，关闭程序...")
            shutdown_event.set()
        finally:
            logger.info("本轮处理结束，等待保存队列清空...")
            local_save_queue.join()
            local_save_queue.put(None)
            saver_thread.join(timeout=10)
            logger.info(f"本轮成功处理 {total_processed_count} 条记录。")

        if shutdown_event.is_set():
            logger.info("程序已中断，退出循环。")
            break
            
        processing_round += 1
        logger.info("等待 5 秒后开始下一轮检查...")
        time.sleep(5)
    # --- MODIFICATION END: 循环结束 ---

    logger.info("所有处理循环已完成，开始最后清理和保存...")
    # --- MODIFICATION: 修正 cleanup_jsonl_file 中的拼写错误 ---
    cleanup_jsonl_file(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, id_column='esid')
    
    try:
        # --- MODIFICATION: 确保即使文件为空也能正常处理 ---
        try:
            judge_output_df = pd.read_json(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, lines=True)
            if not judge_output_df.empty:
                # 只保留成功处理的记录进行最终保存
                judge_output_df = judge_output_df[judge_output_df['status'] == 'success'].copy()
        except (FileNotFoundError, ValueError):
            judge_output_df = pd.DataFrame() # 如果文件不存在或为空，则创建空DataFrame

        # 定义最终列，以防DataFrame为空
        final_cols = ['esid', 'pi_site_name', 'matched_site', 'is_same', 'reason']
        if judge_output_df.empty:
            logger.warning("AI判断步骤没有产出任何成功的结果。")
            judge_output_df = pd.DataFrame(columns=final_cols)
        
        # 确保所有最终列都存在
        for col in final_cols:
            if col not in judge_output_df.columns:
                judge_output_df[col] = None

        judge_output_df = judge_output_df[final_cols] # 保证列的顺序和完整性
        
        judge_output_df.to_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE, index=False)
        logger.info(f"AI判断结果保存至 {config.FINAL_JUDGE_OUTPUT_PARQUET_FILE}")
    except Exception as e:
        logger.error(f"保存最终判断结果时错误: {str(e)}")
        return False

    logger.info("STEP 5 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_5_ai_judgment(config)