#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_0_find_abnormal.py
@Author: WB
@Version: v1.0
@Date: 2025-9-15
@Description: Pre-processing step to classify raw affiliation strings and separate
    abnormal data from valid institution names using an LLM.
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
from openai import OpenAI
from tqdm import tqdm
from config_utils import Config, BATCH_CLASSIFY_PROMPT, shutdown_event, async_saver_local, get_tasks_to_process_generic, cleanup_jsonl_file, logger, sleep_and_retry, limits, retry, wait_exponential


def step_0_find_abnormal(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 0: 正在识别并分离异常机构名......")
    logger.info("="*50)

    # 新增：检查结果文件是否已存在，如果存在则跳过
    if os.path.exists(config.NORMAL_DATA_FILE) and os.path.exists(config.ABNORMAL_DATA_FILE):
        logger.info("结果文件已存在，跳过步骤 0。")
        return True
    
    def read_input_file(file_path_str: str):
        """
        根据文件扩展名智能读取 Parquet 或 Excel 文件。
        """
        try:
            # 根据文件扩展名选择读取方式
            if file_path_str.endswith('.xlsx') or file_path_str.endswith('.xls'):
                logger.info(f"正在以 Excel 格式读取文件: {file_path_str}")
                df = pd.read_excel(file_path_str)
                return df
            elif file_path_str.endswith('.parquet'):
                logger.info(f"正在以 Parquet 格式读取文件: {file_path_str}")
                df = pd.read_parquet(file_path_str)
                return df
            else:
                logger.error(f"不支持的文件格式: {file_path_str}。请上传 .parquet, .xlsx 或 .xls 文件。")
                return None
        except FileNotFoundError:
            logger.error(f"输入文件未找到: {file_path_str}")
            return None
        except Exception as e:
            logger.error(f"读取文件 {file_path_str} 时发生错误: {e}")
            return None
        
    raw_df = read_input_file(str(config.RAW_PARQUET_FILE))
    # 定义列名映射关系
    column_mapping = {
        '_id': 'esid',
        '数据来源': 'data_source',
        '登记号': 'nct_id',
        '研究者单位raw': 'affiliation',
        '研究者机构name(导出名称)': 'pi_site_name',
        '研究机构省份': 'state',
        '研究机构城市': 'city'
    }
    # 只重命名数据框中实际存在的列
    raw_df = raw_df.rename(columns={k: v for k, v in column_mapping.items() if k in raw_df.columns})
    
    if 'affiliation' not in raw_df.columns:
        logger.error(f"输入文件 {config.RAW_PARQUET_FILE} 缺少 'affiliation' 或 '研究者单位raw' 列。")
        return False
            
    # 准备任务
    unique_affiliations_df = raw_df.dropna(subset=['affiliation']).drop_duplicates(subset=['affiliation'], keep='first')
    logger.info(f"找到 {len(unique_affiliations_df)} 个唯一的 affiliation 需要进行分类。")

    tasks, _ = get_tasks_to_process_generic(
        input_df=unique_affiliations_df, 
        id_column='esid',
        output_jsonl_path=config.CLASSIFY_JSONL_FILE ,
        output_id_column='esid'
    )

    @sleep_and_retry
    @limits(calls=config.CALLS_PER_MINUTE, period=config.ONE_MINUTE)
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=20))
    def query_llm_batch(tasks_batch: List[Dict]) -> Dict:
        if shutdown_event.is_set(): 
            return {"classifications": [{"status": "cancelled", **task} for task in tasks_batch]}
        
        user_content_json = classify_content_generator(tasks_batch)

        if not user_content_json: 
            return {"classifications": []}
        try:
            client = OpenAI(
                base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY,
                project=config.API_PROJECT, organization=config.ORGANIZATION, timeout=300.0
            )
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": BATCH_CLASSIFY_PROMPT},
                    {"role": "user", "content": user_content_json}
                ],
                model=config.CLASSIFY_MODEL,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            cleaned_content = re.sub(r'^```json\n|```$', '', content, flags=re.MULTILINE).strip()
            return json.loads(cleaned_content)
        
        except Exception as e:
            logger.error(f"API 调用失败: {str(e)}")
            if "404" in str(e):
                logger.error("API 返回 404 Not Found，可能的模型名称或端点错误")
            raise

    def classify_content_generator(batch: List[Dict]) -> str:
        # 发送给AI的JSON中现在包含 esid
        inputs_for_llm = []
        for task in batch:
            inputs_for_llm.append({
                "esid": task.get("esid"),
                "affiliation": task.get("affiliation")
            })
        return json.dumps({"inputs": inputs_for_llm}, ensure_ascii=False)

    def classify_result_parser(response: Dict, batch: List[Dict]) -> List[Dict]:
        parsed_results = []
        llm_outputs = response.get('classifications', [])
        batch_map = {task.get('esid'): task for task in batch}
        
        for result in llm_outputs:
            result_id = result.get('esid')
            if result_id in batch_map:
                task = batch_map[result_id]
                parsed_results.append({
                    "status": "success",
                    "esid": result_id,
                    "data_source": task.get('data_source'),
                    "nct_id": task.get('nct_id'),
                    "affiliation": task.get('affiliation'),
                    "state": task.get('state'),
                    "city": task.get('city'),
                    "category": result.get("category"),
                    "reason": result.get("reason")
                })
            else:
                parsed_results.append({
                    "status": "failed",
                    "esid": result_id,
                    "data_source": task.get('data_source'),
                    "nct_id": task.get('nct_id'),
                    "affiliation": task.get('affiliation'),
                    "state": task.get('state'),
                    "city": task.get('city'),
                    "category": None,
                    "reason": "LLM响应中缺少此条记录的分类结果。",
                    "error": "LLM response missing"
                })
        return parsed_results

    if not tasks:
        logger.info("所有任务均已处理，无需新操作。")
        return True

    local_save_queue = Queue()
    task_chunks = [tasks[i:i + config.CLASSIFY_BATCH_SIZE] for i in range(0, len(tasks), config.CLASSIFY_BATCH_SIZE)]
    logger.info(f"处理 {len(tasks)} 个任务，分为 {len(task_chunks)} 个批次...")

    saver_thread = threading.Thread(target=async_saver_local, args=(config.CLASSIFY_JSONL_FILE, local_save_queue), daemon=True)
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
                        processed_results = classify_result_parser(llm_response, batch_input)
                        for result in processed_results:
                            local_save_queue.put(result)
                        successful_in_batch = sum(1 for r in processed_results if r.get('status') == 'success')
                        total_processed_count += successful_in_batch
                        pbar.set_description(f"进度 (批次成功: {successful_in_batch}/{len(batch_input)})")
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
    
    cleanup_jsonl_file(config.CLASSIFY_JSONL_FILE, id_column='esid')


    # --- 处理分类结果 ---
    logger.info("所有AI分类任务已完成，正在分离数据...")
    try:
        classification_df = pd.read_json(config.CLASSIFY_JSONL_FILE, lines=True)
        if classification_df.empty:
            logger.info("AI分类结果文件为空，所有原始数据将被视为正常。")
            raw_df.to_parquet(config.NORMAL_DATA_FILE, index=False)
            pd.DataFrame().to_parquet(config.ABNORMAL_DATA_FILE, index=False)
            return True
        
        classification_unique = classification_df.drop_duplicates(subset=['affiliation'], keep='last')
        aff_to_category_map = classification_unique.set_index('affiliation')['category'].to_dict()
        aff_to_reason_map = classification_unique.set_index('affiliation')['reason'].to_dict()
    
        raw_df['category'] = raw_df['affiliation'].map(aff_to_category_map)
        raw_df['reason'] = raw_df['affiliation'].map(aff_to_reason_map)
        
        normal_mask = (raw_df['category'] == 'Valid Institution') | (raw_df['category'].isna())
        normal_data = raw_df[normal_mask].copy()
        abnormal_data = raw_df[~normal_mask].copy()
        normal_data = normal_data.drop(columns=['category'], errors='ignore')

        normal_data.to_parquet(config.NORMAL_DATA_FILE, index=False)
        abnormal_data.to_parquet(config.ABNORMAL_DATA_FILE, index=False)

        logger.info(f"处理完成！")
        logger.info(f"✅ 正常机构数据 ({len(normal_data)})条")
        logger.info(f"❌ 异常数据 ({len(abnormal_data)})条")
        logger.info("STEP 0 完成。")

        return True
    except FileNotFoundError:
        logger.error(f"未找到AI分类结果文件: {config.CLASSIFY_JSONL_FILE}。")
        return False
    except Exception as e:
        logger.error(f"分离数据时发生错误: {e}")
        return False

if __name__ == "__main__":
    config = Config()
    step_0_find_abnormal(config)