#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_2_translate.py
@Author: WB
@Version: v2.0
@Date: 2025-8-27
@Description: Translate unmatched English institution names using OpenAI.
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
from openai import OpenAI
from tqdm import tqdm
from config_utils import Config, BATCH_TRANSLATE_PROMPT, shutdown_event, is_chinese, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential


def step_2_translate_unmatched(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 2: 正在翻译未匹配的英文机构名...")
    logger.info("="*50)

    try:
        unmatched_df = pd.read_parquet(config.UNMATCHED_EN_FILE)
        required_cols = ['original_id', 'affiliation', 'state', 'city']
        if not all(col in unmatched_df.columns for col in required_cols):
            logger.error(f"输入文件缺少必需列: {required_cols}")
            return False
    except FileNotFoundError:
        logger.error(f"未找到输入文件 {config.UNMATCHED_EN_FILE}")
        return False

    tasks, _ = get_tasks_to_process_generic(
        input_df=unmatched_df,
        id_column='original_id',
        output_jsonl_path=config.TRANSLATED_JSONL_FILE,
        output_id_column='original_id'
    )

    @sleep_and_retry
    @limits(calls=config.CALLS_PER_MINUTE, period=config.ONE_MINUTE)
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=20))
    def query_llm_batch(tasks_batch: List[Dict]) -> Dict:
        if shutdown_event.is_set():
            return {"translations": [{"status": "cancelled", **task} for task in tasks_batch]}

        user_content_json = translate_content_generator(tasks_batch)
        if not user_content_json:
            return {"translations": []}

        try:
            # 使用 genai 客户端
            client = genai.Client(
                api_key=config.GENAI_API_KEY,
                http_options=types.HttpOptions(base_url=config.GENAI_BASE_URL)
            )

            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            
            config_genai = types.GenerateContentConfig(
                # tools=[grounding_tool],
                system_instruction=BATCH_TRANSLATE_PROMPT,
                response_modalities=["TEXT"]
            )
            
            response = client.models.generate_content(
                model=config.TRANSLATE_MODEL, # 使用配置中的翻译模型
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
        #     client = OpenAI(
        #         base_url=config.OPENAI_BASE_URL,
        #         api_key=config.OPENAI_API_KEY,
        #         project=config.API_PROJECT,
        #         organization=config.ORGANIZATION,
        #         timeout=300.0
        #     )
        #     response = client.chat.completions.create(
        #         messages=[
        #             {"role": "system", "content": BATCH_TRANSLATE_PROMPT},
        #             {"role": "user", "content": user_content_json}
        #         ],
        #         model=config.TRANSLATE_MODEL,
        #         response_format={"type": "json_object"},
        #         temperature=0.0
        #     )
        #     content = response.choices[0].message.content
        #     cleaned_content = re.sub(r'^```json\n|```$', '', content, flags=re.MULTILINE).strip()
        #     logger.info(f"收到响应: {cleaned_content[:200]}...")
        #     return json.loads(cleaned_content)
        # except Exception as e:
        #     logger.error(f"API 调用失败: {str(e)}")
        #     if "404" in str(e):
        #         logger.error("API 返回 404 Not Found，可能的模型名称或端点错误")
        #     raise

    def translate_content_generator(batch: List[Dict]) -> str:
        tasks_for_llm = []
        for task in batch:
            if not is_chinese(task.get('affiliation', '')):
                tasks_for_llm.append({
                    "original_id": task.get('original_id'),
                    "original_input": task.get('affiliation'),
                    "state": task.get('state', ''),
                    "city": task.get('city', '')
                })
        return json.dumps({"inputs": tasks_for_llm}, ensure_ascii=False) if tasks_for_llm else ""

    def translate_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
        parsed_results = []
        llm_outputs = response.get('translations', [])
        batch_map = {task.get('original_id'): task for task in batch}
        
        for result in llm_outputs:
            result_id = result.get('original_id')
            if result_id in batch_map:
                task = batch_map[result_id]
                parsed_results.append({
                    "status": "success",
                    "original_id": result_id,
                    "original_input": task.get('affiliation'),
                    "original_input_zh": result.get('original_input_zh', ''),
                    "state": task.get('state'),
                    "city": task.get('city')
                })

        llm_processed_ids = {res.get('original_id') for res in parsed_results}
        for task in batch:
            task_id = task.get('original_id')
            if task_id not in llm_processed_ids:
                if is_chinese(task.get('affiliation')):
                    parsed_results.append({
                        "status": "success",
                        "original_id": task_id,
                        "original_input": task.get('affiliation'),
                        "original_input_zh": task.get('affiliation'),
                        "state": task.get('state'),
                        "city": task.get('city')
                    })
                else:
                    parsed_results.append({
                        "status": "failed",
                        "original_id": task_id,
                        "original_input": task.get('affiliation'),
                        "original_input_zh": None,
                        "state": task.get('state'),
                        "city": task.get('city'),
                        "error": "LLM response missing"
                    })
        return parsed_results

    if not tasks:
        logger.info("所有任务均已处理，无需新操作。")
        return True

    local_save_queue = Queue()
    task_chunks = [tasks[i:i + config.TRANSLATE_BATCH_SIZE] for i in range(0, len(tasks), config.TRANSLATE_BATCH_SIZE)]
    logger.info(f"处理 {len(tasks)} 个任务，分为 {len(task_chunks)} 个批次...")

    saver_thread = threading.Thread(target=async_saver_local, args=(config.TRANSLATED_JSONL_FILE, local_save_queue), daemon=True)
    saver_thread.start()

    total_processed_count = 0
    try:
        with tqdm(total=len(task_chunks), desc="翻译进度") as pbar:
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
                        processed_results = translate_result_parser(llm_response, batch_input)
                        for result in processed_results:
                            local_save_queue.put(result)
                        successful_in_batch = sum(1 for r in processed_results if r.get('status') == 'success')
                        total_processed_count += successful_in_batch
                        pbar.set_description(f"翻译进度 (批次成功: {successful_in_batch}/{len(batch_input)})")
                    except Exception as e:
                        logger.error(f"批次处理失败: {str(e)}")
                        for task in batch_input:
                            local_save_queue.put({"status": "failed", "error": str(e), **task})
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
    
    cleanup_jsonl_file(config.TRANSLATED_JSONL_FILE, id_column='original_id')
    logger.info("STEP 2 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_2_translate_unmatched(config)