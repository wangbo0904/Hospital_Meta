#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_5_ai_judgment.py
@Author: WB
@Version: v2.0
@Date: 2025-8-27
@Description: AI judgment of inconsistent institution names using genai.
"""

import pandas as pd
import json
import re
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
from config_utils import Config, BATCH_JUDGE_PROMPT, shutdown_event, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential

def step_5_ai_judgment(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 5: 使用 AI 判断不一致的机构名...")
    logger.info("="*50)

    try:
        judge_input_df = pd.read_parquet(config.JUDGE_INPUT_FILE)
        if judge_input_df.empty:
            logger.info("没有需要AI判断的不一致记录。跳过STEP 5。")
            pd.DataFrame(columns=['original_id', 'pi_site_name', 'matched_site', 'is_same', 'confidence', 'reason']).to_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE, index=False)
            return True
        if 'original_id' not in judge_input_df.columns:
            logger.error(f"输入文件缺少 'original_id' 列")
            return False
    except FileNotFoundError:
        logger.error(f"未找到输入文件 {config.JUDGE_INPUT_FILE}")
        return False

    tasks, _ = get_tasks_to_process_generic(
        input_df=judge_input_df,
        id_column='original_id',
        output_jsonl_path=config.FINAL_JUDGE_OUTPUT_JSONL_FILE,
        output_id_column='original_id'
    )

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
                "original_id": task.get("original_id"),
                "pi_site_name": task.get("pi_site_name", ""),
                "matched_site": task.get("matched_site", "")
            })
        return json.dumps({"inputs": inputs_for_llm}, ensure_ascii=False)

    def judge_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
        batch_map = {task.get("original_id"): task for task in batch}
        llm_outputs = response.get('results', [])
        parsed_results = []
        processed_ids = set()

        for item in llm_outputs:
            result_id = item.get("original_id")
            if result_id in batch_map:
                task = batch_map[result_id]
                parsed_results.append({
                    "status": "success",
                    "original_id": result_id,
                    "pi_site_name": task.get("pi_site_name"),
                    "matched_site": task.get("matched_site"),
                    "is_same": item.get("is_same"),
                    "confidence": item.get("confidence"),
                    "reason": item.get("reason", "")
                })
                processed_ids.add(result_id)
            else:
                logger.warning(f"LLM返回未知ID: {result_id}")

        for task in batch:
            if task['original_id'] not in processed_ids:
                parsed_results.append({
                    "status": "failed",
                    "original_id": task['original_id'],
                    "pi_site_name": task.get("pi_site_name"),
                    "matched_site": task.get("matched_site"),
                    "is_same": False,
                    "confidence": "低",
                    "reason": "LLM响应缺失",
                    "error": "LLM response missing"
                })
        return parsed_results

    if not tasks:
        logger.info("所有任务均已处理，无需新操作。")
        try:
            judge_output_df = pd.read_json(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, lines=True)
            if judge_output_df.empty:
                judge_output_df = pd.DataFrame(columns=['original_id', 'pi_site_name', 'matched_site', 'is_same', 'confidence', 'reason'])
            judge_output_df.to_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE, index=False)
            logger.info(f"AI判断结果保存至 {config.FINAL_JUDGE_OUTPUT_PARQUET_FILE}")
            logger.info("STEP 5 完成。")
            return True
        except Exception as e:
            logger.error(f"保存最终判断结果时错误: {str(e)}")
            return False

    local_save_queue = Queue()
    task_chunks = [tasks[i:i + config.AI_JUDGE_BATCH_SIZE] for i in range(0, len(tasks), config.AI_JUDGE_BATCH_SIZE)]
    logger.info(f"处理 {len(tasks)} 个任务，分为 {len(task_chunks)} 个批次...")

    saver_thread = threading.Thread(target=async_saver_local, args=(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, local_save_queue), daemon=True)
    saver_thread.start()

    total_processed_count = 0
    try:
        with tqdm(total=len(task_chunks), desc="AI判断进度") as pbar:
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
                            local_save_queue.put({
                                "status": "failed",
                                "original_id": task['original_id'],
                                "pi_site_name": task.get("pi_site_name"),
                                "matched_site": task.get("matched_site"),
                                "is_same": False,
                                "confidence": "低",
                                "reason": f"处理失败: {str(e)}",
                                "error": str(e)
                            })
                    pbar.update(1)
    except KeyboardInterrupt:
        logger.info("检测到中断，关闭程序...")
        shutdown_event.set()
    finally:
        logger.info("清理和保存...")
        local_save_queue.join()
        local_save_queue.put(None)
        saver_thread.join(timeout=10)
        logger.info(f"成功处理 {total_processed_count} 条记录。")

    cleanup_jsonl_file(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, id_column='original_id')
    try:
        judge_output_df = pd.read_json(config.FINAL_JUDGE_OUTPUT_JSONL_FILE, lines=True)
        if judge_output_df.empty:
            judge_output_df = pd.DataFrame(columns=['original_id', 'pi_site_name', 'matched_site', 'is_same', 'confidence', 'reason'])
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