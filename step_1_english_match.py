#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_1_english_match.py
@Author: WB
@Version: v2.1
@Date: 2025-09-16
@Description: Initial English name matching for medical institutions.
    Modified to move affiliations with multiple matches to unmatched data.
"""

import pandas as pd
from config_utils import Config, logger
from pypinyin import pinyin, Style
from functools import lru_cache

def to_pinyin_normalized(text: str) -> str:
    """
    将中文文本转换为小写的、无空格的拼音。
    例如："上海市" -> "shanghai"
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.replace('市', '').replace('省', '').replace('区', '').replace('县', '')
    pinyin_list = pinyin(text, style=Style.NORMAL)
    return "".join(item[0] for item in pinyin_list).lower()

def step_1_initial_english_match(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 1: 正在执行初始英文机构名匹配...")
    logger.info("="*50)

    def read_input_file(file_path_str: str):
        """
        根据文件扩展名智能读取 Parquet 或 Excel 文件。
        """
        try:
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

    # 读取字典和输入数据
    site_dict_df = read_input_file(str(config.SITE_DICT_FILE))
    site_test = pd.read_parquet(config.NORMAL_DATA_FILE)

    if site_dict_df is None or site_test is None:
        logger.error("一个或多个输入文件无法读取。流程中止。")
        return False
    
    # 去重输入数据，基于 esid
    site_test = site_test.drop_duplicates(subset=['esid'], keep='first')
    logger.info(f"去重后输入数据量: {len(site_test)}")

    # 验证必需列
    required_cols = ['esid', 'data_source', 'nct_id', 'affiliation', 'pi_site_name', 'state', 'city']
    if not all(col in site_test.columns for col in required_cols):
        logger.error(f"原始数据缺少必需列: {required_cols}")
        logger.error(f"当前列: {site_test.columns.tolist()}")
        return False

    # 清理 affiliation
    site_test['affiliation_clean'] = site_test['affiliation'].astype(str).str.strip().str.lower()

    # 准备字典
    site_dict_df = site_dict_df.rename(columns={'city': 'hospital_city'})
    site_dict_df['hospital_name_en_clean'] = site_dict_df['hospital_name_en'].astype(str).str.strip().str.lower()
    site_dict_df['province_pinyin'] = site_dict_df['province'].apply(to_pinyin_normalized)
    site_dict_df['hospital_city_pinyin'] = site_dict_df['hospital_city'].apply(to_pinyin_normalized)
    site_dict_df = site_dict_df.drop_duplicates(subset=['hospital_name_en_clean'])
    logger.info(f"字典唯一 hospital_name_en_clean: {len(site_dict_df)}")

    # 缓存匹配结果
    @lru_cache(maxsize=None)
    def find_match(affiliation_clean):
        match = site_dict_df[site_dict_df['hospital_name_en_clean'].str.contains(affiliation_clean, case=False, na=False, regex=False)]
        return match

    # 第一轮匹配，记录多匹配的 esid
    initial_matches = []
    multiple_match_esids = set()  # 记录匹配多条的 esid
    for _, row in site_test.iterrows():
        affiliation_clean = row['affiliation_clean']
        match = find_match(affiliation_clean)
        if not match.empty:
            if len(match) > 1:
                # 如果匹配多条，记录 esid，放入未匹配
                multiple_match_esids.add(row['esid'])
                logger.info(f"多匹配: esid={row['esid']}, affiliation={row['affiliation']}, 匹配数={len(match)}")
            else:
                # 单一匹配，加入 initial_matches
                initial_matches.append(match.copy().assign(
                    esid=row['esid'], 
                    data_source=row['data_source'], 
                    nct_id=row['nct_id'], 
                    affiliation=row['affiliation'], 
                    pi_site_name=row['pi_site_name'], 
                    state=row['state'], 
                    city=row['city']
                ))

    if initial_matches:
        initial_matched = pd.concat(initial_matches).drop_duplicates(subset=['esid'], keep='first')
    else:
        initial_matched = pd.DataFrame()
    logger.info(f"初步匹配记录数: {len(initial_matched)}, 唯一 esid: {initial_matched['esid'].nunique()}")

    # 第二轮地理位置验证
    initial_matched['state_norm'] = initial_matched['state'].apply(to_pinyin_normalized)
    initial_matched['city_norm'] = initial_matched['city'].apply(to_pinyin_normalized)

    cond_state_match = initial_matched.apply(lambda row: row['state_norm'] in row['province_pinyin'] or row['province_pinyin'] in row['state_norm'], axis=1)
    cond_city_match = initial_matched.apply(lambda row: row['city_norm'] in row['hospital_city_pinyin'] or row['hospital_city_pinyin'] in row['city_norm'], axis=1)

    fully_matched = initial_matched[cond_state_match & cond_city_match]
    fully_matched['confidence'] = 'name_en匹配'

    # 未匹配 = 原始 - 匹配 + 多匹配的记录
    unmatched = site_test[site_test['esid'].isin(multiple_match_esids) | ~site_test['esid'].isin(fully_matched['esid'])]

    # 保存
    matched_cols_to_save = ['esid', 'data_source', 'nct_id', 'affiliation', 'state', 'city', 'pi_site_name', 'hospital_name', 'hospital_name_en', 'province', 'hospital_city', 'confidence']
    matched_to_save = fully_matched[matched_cols_to_save].rename(columns={
        'hospital_name': 'matched_site',
        'hospital_name_en': 'matched_site_en',
        'province': 'matched_site_province',
        'hospital_city': 'matched_site_city'
    })
    matched_to_save.to_parquet(config.MATCHED_EN_FILE, index=False)

    unmatched_cols_to_save = ['esid', 'data_source', 'nct_id', 'affiliation', 'pi_site_name', 'state', 'city']
    unmatched[unmatched_cols_to_save].to_parquet(config.UNMATCHED_EN_FILE, index=False)

    logger.info(f"【最终结果】英文名和地理位置双重验证成功 {len(matched_to_save)} 条记录。")
    logger.info(f"【最终结果】未匹配或仅名称匹配 {len(unmatched)} 条记录。")
    logger.info(f"总记录数: {len(matched_to_save) + len(unmatched)}")
    logger.info("STEP 1 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_1_initial_english_match(config)