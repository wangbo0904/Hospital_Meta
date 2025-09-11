#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_1_english_match.py
@Author: WB
@Version: v2.0
@Date: 2025-8-27
@Description: Initial English name matching for medical institutions.
"""

import pandas as pd
import sys
sys.path.append('C:/Users/YYMF/PythonProjects/PI&Site/SITE_APP')
from config_utils import Config, logger
from pypinyin import pinyin, Style

def to_pinyin_normalized(text: str) -> str:
    """
    将中文文本转换为小写的、无空格的拼音。
    例如："上海市" -> "shanghai"
    """
    if not isinstance(text, str) or not text:
        return ""
    # 去掉常见的后缀并转换为拼音
    text = text.replace('市', '').replace('省', '').replace('区', '').replace('县', '')
    pinyin_list = pinyin(text, style=Style.NORMAL)
    return "".join(item[0] for item in pinyin_list).lower()

def step_1_initial_english_match(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 1: 正在执行初始英文机构名匹配...")
    logger.info("="*50)

    try:
        site_dict_df = pd.read_parquet(config.SITE_DICT_FILE)
        site_test = pd.read_parquet(config.RAW_PARQUET_FILE)
    except FileNotFoundError as e:
        logger.error(f"输入文件未找到: {e}")
        return False
    
    site_test = site_test.reset_index().rename(columns={'index': 'original_id'})
    print(f"已为 {len(site_test)} 条记录添加 'original_id'。")

    required_cols = ['original_id', 'affiliation', 'pi_site_name', 'state', 'city']
    if not all(col in site_test.columns for col in required_cols):
        print(f"原始数据缺少必需列: {required_cols}")
        print(f"当前列: {site_test.columns.tolist()}")
        return False

    site_test['affiliation_clean'] = site_test['affiliation'].astype(str).str.strip().str.lower()
    site_test_deduped = site_test.drop_duplicates(subset=['affiliation_clean'], keep='first').copy()
    print(f"去重后剩余 {len(site_test_deduped)} 条记录进行匹配。")

    site_dict_df['hospital_name_en_clean'] = site_dict_df['hospital_name_en'].str.strip().str.lower()
    site_dict_df_deduped = site_dict_df.drop_duplicates(subset=['hospital_name_en_clean']).copy()

    # 重命名site_dict_df的city列，避免冲突
    site_dict_df_deduped = site_dict_df_deduped.rename(columns={'city': 'hospital_city'})

    # --- 2. 第一轮：仅基于英文名称进行匹配 ---
    initial_matched = pd.merge(
        site_test_deduped,
        site_dict_df_deduped,
        left_on='affiliation_clean',
        right_on='hospital_name_en_clean',
        how='inner'
    )
    print(f"初步英文名匹配到 {len(initial_matched)} 条记录，开始进行地理位置验证...")

    # --- 3. 第二轮：在初步匹配结果上，进行地理位置交叉验证 ---
    
    # 准备用于比较的拼音列
    # 对输入的 state/city 进行归一化 (小写, 去空格)
    initial_matched['state_norm'] = initial_matched['state'].apply(to_pinyin_normalized).str.lower().str.replace(' ', '')
    initial_matched['city_norm'] = initial_matched['city'].apply(to_pinyin_normalized).str.lower().str.replace(' ', '')
    
    # 对字典中的 province/city 进行归一化 (转拼音)
    initial_matched['province_pinyin'] = initial_matched['province'].apply(to_pinyin_normalized).str.lower().str.replace(' ', '')
    initial_matched['hospital_city_pinyin'] = initial_matched['hospital_city'].apply(to_pinyin_normalized).str.lower().str.replace(' ', '')

    # 定义匹配条件
    # 条件1: 省份匹配 (输入的state拼音 出现在 字典的province拼音中，反之亦然)
    cond_state_match = initial_matched.apply(
        lambda row: row['state_norm'] in row['province_pinyin'] or row['province_pinyin'] in row['state_norm'], 
        axis=1
    )
    # 条件2: 城市匹配 (逻辑同上)
    cond_city_match = initial_matched.apply(
        lambda row: row['city_norm'] in row['hospital_city_pinyin'] or row['hospital_city_pinyin'] in row['city_norm'], 
        axis=1
    )

    # 最终的精确匹配结果，是同时满足省份和城市匹配的记录
    # 注意：这里我们使用 `&` (与) 逻辑，要求两者都匹配。如果想放宽，可以用 `|` (或)
    fully_matched = initial_matched[cond_state_match & cond_city_match]
    fully_matched['confidence'] = 'name_en匹配'
    print(f"地理位置验证通过的有 {len(fully_matched)} 条记录。")

    # --- 4. 确定最终的未匹配列表 ---
    # 未匹配 = 原始数据 - 完全匹配的
    unmatched = site_test_deduped[~site_test_deduped['affiliation_clean'].isin(fully_matched['affiliation_clean'])]

    # --- 5. 保存结果 ---
    matched_cols_to_save = ['original_id', 'affiliation', 'state', 'city', 'pi_site_name', 'hospital_name', 'hospital_name_en', 'province', 'hospital_city', 'confidence']
    matched_to_save = fully_matched[matched_cols_to_save].rename(columns={
        'hospital_name': 'matched_site',
        'hospital_name_en': 'matched_site_en',
        'province': 'matched_site_province',
        'hospital_city': 'matched_site_city'
    })
    matched_to_save.to_parquet(config.MATCHED_EN_FILE, index=False)

    unmatched_cols_to_save = ['original_id', 'affiliation', 'pi_site_name', 'state', 'city']
    unmatched[unmatched_cols_to_save].to_parquet(config.UNMATCHED_EN_FILE, index=False)

    logger.info(f"【最终结果】英文名和地理位置双重验证成功 {len(matched_to_save)} 条记录。")
    logger.info(f"【最终结果】未匹配或仅名称匹配 {len(unmatched)} 条记录。")
    logger.info("STEP 1 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_1_initial_english_match(config)