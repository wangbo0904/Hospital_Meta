#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_4_ai_candidate_selection.py
@Author: WB
@Version: v2.0
@Date: 2025-8-27
@Description: AI candidate selection using genai.
"""

import pandas as pd
import json
import re
import threading
import numpy as np
from google import genai
from queue import Queue
from google.genai import types
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import stop_after_attempt
from tqdm import tqdm
import sys
sys.path.append('C:/Users/YYMF/PythonProjects/PI&Site/SITE_APP')
from config_utils import Config, BATCH_AI_SELECT_PROMPT, shutdown_event, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential

def step_4_ai_candidate_selection(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 4: 使用 AI 从候选列表中选择最佳匹配...")
    logger.info("="*50)

    try:
        ai_input_df = pd.read_parquet(config.AI_META_INPUT_FILE)
        if ai_input_df.empty:
            logger.info("没有需要AI处理的模糊匹配记录。跳过STEP 4。")
            pd.DataFrame(columns=['original_id', 'affiliation', 'affiliation_cn', 'state', 'city', 'matched_site', 'matched_site_en', 'matched_site_province', 'matched_site_city', 'reason']).to_parquet(config.ALL_MATCHES_COMBINED_FILE, index=False)
            return True
    except FileNotFoundError:
        logger.error(f"未找到输入文件 {config.AI_META_INPUT_FILE}")
        return False

    def clean_candidates_list(candidates):
        if not isinstance(candidates, (list, np.ndarray)):
            return []
        cleaned_list = []
        for cand in candidates:
            if isinstance(cand, dict):
                alias_list = cand.get('hospital_alias')
                if isinstance(alias_list, np.ndarray):
                    alias_list = alias_list.tolist()
                cleaned_list.append({
                    'hospital_name': cand.get('hospital_name'),
                    'hospital_name_en': cand.get('hospital_name_en'),
                    'hospital_alias': alias_list,
                    'hospital_province': cand.get('hospital_province'),
                    'hospital_city': cand.get('hospital_city')
                })
        return cleaned_list

    ai_input_df['candidates'] = ai_input_df['candidates'].apply(clean_candidates_list)
    tasks, _ = get_tasks_to_process_generic(
        input_df=ai_input_df,
        id_column='original_id',
        output_jsonl_path=config.AI_META_OUTPUT_JSONL_FILE,
        output_id_column='original_id'
    )

    @sleep_and_retry
    @limits(calls=config.CALLS_PER_MINUTE, period=config.ONE_MINUTE)
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=20))
    def query_llm_batch(tasks_batch: List[Dict]) -> Dict:
        if shutdown_event.is_set():
            return {"results": [{"status": "cancelled", **task} for task in tasks_batch]}

        user_content_json = select_content_generator(tasks_batch)
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
                system_instruction=BATCH_AI_SELECT_PROMPT,
                response_modalities=["TEXT"]
            )
            response = client.models.generate_content(
                model=config.AI_SELECT_MODEL,
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

    def select_content_generator(batch: List[Dict]) -> str:
        inputs_for_llm = []
        for task in batch:
            inputs_for_llm.append({
                "original_id": task.get("original_id"),
                "affiliation": task.get("affiliation"),
                "affiliation_cn": task.get("affiliation_cn"),
                "state": task.get("state"),
                "city": task.get("city"),
                "candidates": task.get("candidates")
            })
        return json.dumps({"inputs": inputs_for_llm}, ensure_ascii=False)

    def select_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
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
                    "affiliation": task.get("affiliation"),
                    "affiliation_cn": task.get("affiliation_cn"),
                    "state": task.get("state"),
                    "city": task.get("city"),
                    "matched_site": item.get("matched_site"),
                    "matched_site_en": item.get("matched_site_en"),
                    "matched_site_province": item.get("matched_province"),
                    "matched_site_city": item.get("matched_city"),
                    "confidence": item.get("confidence"),
                    "reason_tag": item.get("reason_tag"),
                    "reason": item.get("reason")
                })
                processed_ids.add(result_id)
            else:
                logger.warning(f"LLM返回未知ID: {result_id}")

        for task in batch:
            if task['original_id'] not in processed_ids:
                parsed_results.append({"status": "failed", "error": "LLM response missing", **task})
        return parsed_results

    if not tasks:
        logger.info("所有任务均已处理，无需新操作。")
    else:
        local_save_queue = Queue()
        task_chunks = [tasks[i:i + config.AI_SELECT_BATCH_SIZE] for i in range(0, len(tasks), config.AI_SELECT_BATCH_SIZE)]
        logger.info(f"处理 {len(tasks)} 个任务，分为 {len(task_chunks)} 个批次...")

        saver_thread = threading.Thread(target=async_saver_local, args=(config.AI_META_OUTPUT_JSONL_FILE, local_save_queue), daemon=True)
        saver_thread.start()

        total_processed_count = 0
        try:
            with tqdm(total=len(task_chunks), desc="AI选择进度") as pbar:
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
                            processed_results = select_result_parser(llm_response, batch_input)
                            for result in processed_results:
                                local_save_queue.put(result)
                            successful_in_batch = sum(1 for r in processed_results if r.get('status') == 'success')
                            total_processed_count += successful_in_batch
                            pbar.set_description(f"AI选择进度 (批次成功: {successful_in_batch}/{len(batch_input)})")
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

    cleanup_jsonl_file(config.AI_META_OUTPUT_JSONL_FILE, id_column='original_id')
    try:
        ai_matched_df = pd.read_json(config.AI_META_OUTPUT_JSONL_FILE, lines=True)
        if ai_matched_df.empty:
            ai_matched_df = pd.DataFrame(columns=['original_id', 'affiliation', 'affiliation_cn', 'state', 'city', 'matched_site', 'matched_site_en', 'matched_site_province', 'matched_site_city', 'reason'])

        unmatched_en_df = pd.read_parquet(config.UNMATCHED_EN_FILE)
        matched_en_df = pd.read_parquet(config.MATCHED_EN_FILE)
        meta_matched_df = pd.read_parquet(config.META_PARTIAL_EXACT_MATCHED_FILE)

        id_to_pi_site_map = pd.concat([
            matched_en_df[['original_id', 'pi_site_name']],
            unmatched_en_df[['original_id', 'pi_site_name']]
        ]).drop_duplicates(subset=['original_id']).set_index('original_id')['pi_site_name'].to_dict()

        ai_matched_df['pi_site_name'] = ai_matched_df['original_id'].map(id_to_pi_site_map)
        meta_matched_df['pi_site_name'] = meta_matched_df['original_id'].map(id_to_pi_site_map)

        def ensure_columns(df, cols):
            for col in cols:
                if col not in df.columns:
                    df[col] = None
            return df[cols]

        cols_ai = ['original_id', 'affiliation', 'affiliation_cn', 'state', 'city', 'pi_site_name', 'matched_site', 'matched_site_en', 'matched_site_province', 'matched_site_city', 'confidence', 'reason_tag', 'reason']
        cols_en = ['original_id', 'affiliation', 'state', 'city', 'pi_site_name', 'matched_site', 'matched_site_en', 'matched_site_province', 'matched_site_city', 'confidence']
        cols_meta = ['original_id', 'affiliation', 'affiliation_cn', 'state', 'city', 'pi_site_name', 'matched_site', 'matched_site_en', 'matched_site_province', 'matched_site_city', 'confidence']

        final_results = pd.concat([
            ensure_columns(ai_matched_df, cols_ai),
            ensure_columns(matched_en_df, cols_en),
            ensure_columns(meta_matched_df, cols_meta)
        ], ignore_index=True)

        final_results.to_parquet(config.ALL_MATCHES_COMBINED_FILE, index=False)
        logger.info(f"整合 {len(final_results)} 条匹配结果，保存至 {config.ALL_MATCHES_COMBINED_FILE}")

        diff_sites = final_results[
            (final_results['pi_site_name'] != final_results['matched_site']) &
            (final_results['pi_site_name'].notna()) & (final_results['matched_site'].notna()) &
            (final_results['pi_site_name'] != '') & (final_results['matched_site'] != '')
        ].copy()

        diff_sites.to_parquet(config.JUDGE_INPUT_FILE, index=False)
        logger.info(f"找到 {len(diff_sites)} 条不一致记录，保存至 {config.JUDGE_INPUT_FILE}")
    except Exception as e:
        logger.error(f"整合结果时错误: {str(e)}")
        return False

    logger.info("STEP 4 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_4_ai_candidate_selection(config)