#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_3_candidate_matching.py
@Author: WB
@Version: v2.0
@Date: 2025-8-27
@Description: Perform exact, partial, and fuzzy matching for translated Chinese names.
"""

import pandas as pd
import json
import re
import numpy as np
from rapidfuzz import process, fuzz
from functools import lru_cache
from tqdm import tqdm
from config_utils import Config, logger

def step_3_candidate_matching(config: Config):
    logger.info("\n" + "="*50)
    logger.info("STEP 3: 正在对翻译后的中文名进行候选匹配...")
    logger.info("="*50)

    try:
        data = []
        with open(config.TRANSLATED_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                if json_obj.get('status') == 'success':
                    data.append({
                        'original_id': json_obj['original_id'],
                        'affiliation': json_obj['original_input'],
                        'affiliation_cn': json_obj['original_input_zh'],
                        'state': json_obj.get('state', ''),
                        'city': json_obj.get('city', '')
                    })
        translated_df = pd.DataFrame(data)
        translated_df['affiliation_cn'] = translated_df['affiliation_cn'].astype(str).str.replace(r'\s+', '', regex=True)
        site_dict_df = pd.read_parquet(config.SITE_DICT_FILE)
    except FileNotFoundError as e:
        logger.error(f"输入文件未找到: {e}")
        return False

    def query_hospitals(site_df, translated_df_input, output_file):
        FINAL_CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
        INITIAL_FETCH_MULTIPLIER = 5
        INITIAL_FETCH_LIMIT = FINAL_CANDIDATE_LIMIT * INITIAL_FETCH_MULTIPLIER

        required_columns = ['hospital_name', 'hospital_name_en', 'hospital_alias', 'province', 'city']
        if not all(col in site_df.columns for col in required_columns):
            raise ValueError(f"机构字典缺少列: {required_columns}")

        result_dict = {
            "exact_matches": {}, "partial_matches": {},
            "fuzzy_matches": {}, "unmatched_queries": []
        }

        logger.info("整合机构字典...")
        consolidated_candidates = {}
        for _, row in tqdm(site_df.iterrows(), total=len(site_df), desc="整合候选机构"):
            name = row['hospital_name'].strip()
            if name not in consolidated_candidates:
                aliases = set()
                alias_str = row['hospital_alias']
                if isinstance(alias_str, str):
                    if '\n' in alias_str:
                        alias_list = [a.strip() for a in alias_str.split('\n') if a.strip()]
                    elif ',' in alias_str:
                        alias_list = [a.strip() for a in alias_str.split(',') if a.strip()]
                    else:
                        alias_list = [alias_str.strip()]
                    aliases.update(alias_list)
                consolidated_candidates[name] = {
                    'name_en': row.get('hospital_name_en', ''),
                    'aliases': list(aliases),
                    'province': row.get('province', ''),
                    'city': row.get('city', '')
                }

        logger.info("进行精确匹配...")
        matched_ids = set()
        for _, row in tqdm(translated_df_input.iterrows(), total=len(translated_df_input), desc="精确匹配中"):
            original_id = row['original_id']
            original_query = row['affiliation_cn']
            query_lower = original_query.lower()
            exact_match_found = False
            for hospital_name, data in consolidated_candidates.items():
                if query_lower == hospital_name.lower():
                    match_info = {
                        "original_id": original_id,
                        "matched_text": hospital_name,
                        "hospital_name": hospital_name,
                        "hospital_name_en": data['name_en'],
                        "hospital_alias": data['aliases'],
                        "hospital_province": data['province'],
                        "hospital_city": data['city'],
                        "match_type": "exact_name"
                    }
                    result_dict.setdefault("exact_matches", {}).setdefault(original_query, []).append(match_info)
                    matched_ids.add(original_id)
                    exact_match_found = True
                    break
                for alias in data['aliases']:
                    if query_lower == str(alias).lower():
                        match_info = {
                            "original_id": original_id,
                            "matched_text": alias,
                            "hospital_name": hospital_name,
                            "hospital_name_en": data['name_en'],
                            "hospital_alias": data['aliases'],
                            "hospital_province": data['province'],
                            "hospital_city": data['city'],
                            "match_type": "exact_alias"
                        }
                        result_dict.setdefault("exact_matches", {}).setdefault(original_query, []).append(match_info)
                        matched_ids.add(original_id)
                        exact_match_found = True
                        break
                if exact_match_found:
                    break

        logger.info("进行部分匹配...")
        hospital_lookup_map = {}
        for hospital_name, data in consolidated_candidates.items():
            name_lower = hospital_name.lower()
            if name_lower not in hospital_lookup_map:
                hospital_lookup_map[name_lower] = []
            hospital_lookup_map[name_lower].append({
                "is_main_name": True,
                "hospital_name": hospital_name,
                "data": data
            })
            for alias in data['aliases']:
                alias_lower = str(alias).lower()
                if alias_lower not in hospital_lookup_map:
                    hospital_lookup_map[alias_lower] = []
                hospital_lookup_map[alias_lower].append({
                    "is_main_name": False,
                    "hospital_name": hospital_name,
                    "data": data
                })

        normalized_lookup_map = {}
        for hospital_name, data in consolidated_candidates.items():
            norm_name = re.sub(r'[省市区县]', '', hospital_name).lower()
            if norm_name not in normalized_lookup_map:
                normalized_lookup_map[norm_name] = []
            normalized_lookup_map[norm_name].append({
                "is_main_name": True,
                "hospital_name": hospital_name,
                "data": data
            })
            for alias in data['aliases']:
                norm_alias = re.sub(r'[省市区县]', '', str(alias)).lower()
                if norm_alias not in normalized_lookup_map:
                    normalized_lookup_map[norm_alias] = []
                normalized_lookup_map[norm_alias].append({
                    "is_main_name": False,
                    "hospital_name": hospital_name,
                    "data": data
                })

        def normalize_order(text: str) -> str:
            match = re.search(r'(第[一二三四五六七八九十\d]+).*?(附属)|(附属).*?(第[一二三四五六七八九十\d]+)', text)
            if match:
                num_part = match.group(1) or match.group(4)
                affix_part = "附属"
                temp_text = text.replace(num_part, "TEMP_NUM").replace(affix_part, "TEMP_AFFIX")
                return temp_text.replace("TEMP_NUM", num_part).replace("TEMP_AFFIX", affix_part)
            return text

        df_for_partial = translated_df_input[~translated_df_input['original_id'].isin(matched_ids)]
        for _, row in tqdm(df_for_partial.iterrows(), total=len(df_for_partial), desc="部分匹配中"):
            original_id = row['original_id']
            original_query = row['affiliation_cn']
            query_lower = original_query.lower()
            best_match_info = None
            matched_text = ""
            partial_type = ""

            def find_closest_length_match(query, candidates):
                if not candidates:
                    return None
                return min(candidates, key=lambda c: abs(len(query) - len(c['hospital_name'])))

            bracket_match = re.match(r'(.+?)\s*[\(（](.+)[\)）]', original_query)
            if bracket_match:
                main_name, bracket_name = map(str.strip, bracket_match.groups())
                main_matches = hospital_lookup_map.get(main_name.lower())
                if main_matches:
                    best_match_info = find_closest_length_match(main_name, main_matches)
                    if best_match_info:
                        matched_text = main_name
                        partial_type = "bracket_main"
                else:
                    bracket_matches = hospital_lookup_map.get(bracket_name.lower())
                    if bracket_matches:
                        best_match_info = find_closest_length_match(bracket_name, bracket_matches)
                        if best_match_info:
                            matched_text = bracket_name
                            partial_type = "bracket_content"

            if not best_match_info:
                match = re.search(r"(.+(?:医院|院区|分院|医疗区|医学中心))", original_query)
                if match:
                    core_name = match.group(1).strip()
                    possible_matches = hospital_lookup_map.get(core_name.lower())
                    if possible_matches:
                        best_match_info = find_closest_length_match(core_name, possible_matches)
                        if best_match_info:
                            matched_text = best_match_info['hospital_name'] if best_match_info['is_main_name'] else core_name
                            partial_type = "department"

            if not best_match_info:
                normalized_query_region = re.sub(r'[省市区县]', '', original_query).lower()
                possible_matches_region = normalized_lookup_map.get(normalized_query_region)
                if possible_matches_region:
                    best_match_info = find_closest_length_match(normalized_query_region, possible_matches_region)
                    if best_match_info:
                        aliases = best_match_info['data']['aliases']
                        matched_text = best_match_info['hospital_name'] if best_match_info['is_main_name'] else aliases[0] if aliases else best_match_info['hospital_name']
                        partial_type = "region_normalized"

            if not best_match_info:
                normalized_query = normalize_order(original_query)
                if normalized_query != original_query:
                    for hospital_name, data in consolidated_candidates.items():
                        normalized_candidate = normalize_order(hospital_name)
                        if normalized_query == normalized_candidate:
                            best_match_info = hospital_lookup_map.get(hospital_name.lower())[0]
                            matched_text = original_query
                            partial_type = "order_normalized"
                            break
                    if not best_match_info:
                        for hospital_name, data in consolidated_candidates.items():
                            for alias in data['aliases']:
                                normalized_alias = normalize_order(str(alias))
                                if normalized_query == normalized_alias:
                                    best_match_info = hospital_lookup_map.get(str(alias).lower())[0]
                                    matched_text = original_query
                                    partial_type = "order_normalized_alias"
                                    break
                            if best_match_info:
                                break

            if best_match_info:
                source_data = best_match_info['data']
                match_info = {
                    "original_id": original_id,
                    "matched_text": matched_text,
                    "hospital_name": best_match_info['hospital_name'],
                    "hospital_name_en": source_data['name_en'],
                    "hospital_alias": source_data['aliases'],
                    "hospital_province": source_data['province'],
                    "hospital_city": source_data['city'],
                    "match_type": "partial",
                    "partial_type": partial_type
                }
                result_dict.setdefault("partial_matches", {}).setdefault(original_query, []).append(match_info)
                matched_ids.add(original_id)

        logger.info("准备模糊匹配...")
        df_for_fuzzy = translated_df_input[~translated_df_input['original_id'].isin(matched_ids)]
        if not df_for_fuzzy.empty:
            @lru_cache(maxsize=None)
            def extract_location(text: str) -> list:
                text = str(text)
                locations = re.findall(r'(\w{2,5}(?:省|市|区|县))', text)
                level_map = {'省': 1, '市': 2, '区': 3, '县': 3}
                return sorted(locations, key=lambda x: level_map.get(x[-1], 4))

            @lru_cache(maxsize=None)
            def extract_number_code(text: str) -> set:
                full_width = "０１２３４５６７８９"
                half_width = "0123456789"
                translation_table = str.maketrans(full_width, half_width)
                normalized_text = text.translate(translation_table)
                num_map = {
                    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                    '六': '6', '七': '7', '八': '8', '九': '9', '零': '0', '〇': '0', 'O': '0', 'o': '0', '○': '0'
                }
                pattern = re.compile('|'.join(re.escape(key) for key in num_map.keys()))
                normalized_text = pattern.sub(lambda m: num_map[m.group(0)], normalized_text)
                codes = re.findall(r'\d{2,4}', normalized_text)
                return set(codes)

            logger.info("为候选机构预处理...")
            text_to_info_map = {}
            for name, data in tqdm(consolidated_candidates.items(), desc="预处理候选"):
                all_codes = extract_number_code(name)
                for alias in data['aliases']:
                    all_codes.update(extract_number_code(str(alias)))
                info = {
                    "main_name": name,
                    "main_name_en": data['name_en'],
                    "aliases": data['aliases'],
                    "province": data['province'],
                    "city": data['city'],
                    "codes": all_codes
                }
                text_to_info_map[name] = info
                text_to_info_map[data['name_en']] = info
                for alias in data['aliases']:
                    text_to_info_map[str(alias)] = info
                text_to_info_map[data['province']] = info
                text_to_info_map[data['city']] = info

            candidate_texts = list(text_to_info_map.keys())

            @lru_cache(maxsize=100000)
            def hospital_scorer(query_tuple, choice, *, score_cutoff=0):
                query, state, city = query_tuple
                choice_info = text_to_info_map.get(choice)
                if not choice_info:
                    return 0

                if query in choice or choice in query:
                    return 95 + len(min(query, choice, key=len)) / len(max(query, choice, key=len)) * 5
                query_core_match = re.search(r'([^\d\(\)（）\s]+?医院)', query)
                if query_core_match:
                    query_core = query_core_match.group(1)
                    if query_core in choice or choice in query_core:
                        return 90

                query_codes = extract_number_code(query)
                choice_codes = choice_info['codes']
                code_score = 0
                if query_codes and choice_codes:
                    if query_codes == choice_codes:
                        code_score = 100
                    elif query_codes & choice_codes:
                        code_score = 80
                    else:
                        code_score = 30
                elif query_codes or choice_codes:
                    code_score = 30
                else:
                    code_score = 85

                name_score = fuzz.WRatio(query, choice)
                loc_score = 0
                query_locs = extract_location(query)
                choice_locs = extract_location(choice)
                if query_locs and choice_locs:
                    q_loc_last = query_locs[-1]
                    c_loc_last = choice_locs[-1]
                    q_suffix = q_loc_last[-1]
                    c_suffix = c_loc_last[-1]
                    if q_suffix == c_suffix:
                        if q_loc_last == c_loc_last:
                            loc_score = 100
                        else:
                            loc_score = 10
                    else:
                        loc_score = 50
                elif not query_locs and not choice_locs:
                    loc_score = 70
                else:
                    loc_score = 50

                WEIGHTS = {'code': 0.40, 'location': 0.10, 'name_similarity': 0.50}
                final_score = (code_score * WEIGHTS['code'] + loc_score * WEIGHTS['location'] + name_score * WEIGHTS['name_similarity'])
                if final_score < score_cutoff:
                    return 0
                return final_score

            for _, row in tqdm(df_for_fuzzy.iterrows(), total=len(df_for_fuzzy), desc="模糊匹配中"):
                original_id = row['original_id']
                original_query = row['affiliation_cn']
                state = row['state']
                city = row['city']
                query_tuple_for_scorer = (original_query, state, city)
                matches = process.extract(
                    query_tuple_for_scorer,
                    candidate_texts,
                    scorer=hospital_scorer,
                    limit=INITIAL_FETCH_LIMIT
                )
                fuzzy_results = []
                seen_hospital_names = set()
                for match_text, score, _ in matches:
                    if len(fuzzy_results) >= FINAL_CANDIDATE_LIMIT:
                        break
                    source_info = text_to_info_map[match_text]
                    hospital_name_key = source_info['main_name']
                    if hospital_name_key not in seen_hospital_names:
                        fuzzy_results.append({
                            "original_id": original_id,
                            "matched_text": match_text,
                            "hospital_name": hospital_name_key,
                            "hospital_name_en": source_info['main_name_en'],
                            "hospital_alias": source_info['aliases'],
                            "hospital_province": source_info['province'],
                            "hospital_city": source_info['city'],
                            "match_type": "fuzzy",
                            "similarity_score": round(score, 2)
                        })
                        seen_hospital_names.add(hospital_name_key)
                if fuzzy_results:
                    result_dict.setdefault("fuzzy_matches", {}).setdefault(original_query, []).extend(fuzzy_results)

        exact_ids = {m['original_id'] for matches in result_dict.get('exact_matches', {}).values() for m in matches}
        partial_ids = {m['original_id'] for matches in result_dict.get('partial_matches', {}).values() for m in matches}
        fuzzy_ids = {m['original_id'] for matches in result_dict.get('fuzzy_matches', {}).values() for m in matches}
        partial_ids -= exact_ids
        fuzzy_ids -= exact_ids
        fuzzy_ids -= partial_ids

        exact_count = len(exact_ids)
        partial_count = len(partial_ids)
        fuzzy_count = len(fuzzy_ids)
        all_input_ids = set(translated_df_input['original_id'])
        final_all_matched_ids = exact_ids | partial_ids | fuzzy_ids
        unmatched_ids = all_input_ids - final_all_matched_ids
        unmatched_count = len(unmatched_ids)

        id_to_query_map = pd.Series(translated_df_input.affiliation_cn.values, index=translated_df_input.original_id).to_dict()
        result_dict["unmatched_queries"] = [
            {"original_id": oid, "query": id_to_query_map.get(oid, "未知查询")}
            for oid in unmatched_ids
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"匹配结果保存至 {output_file}")
        logger.info(f"精确匹配: {exact_count} 条记录")
        logger.info(f"部分匹配: {partial_count} 条记录")
        logger.info(f"模糊匹配: {fuzzy_count} 条记录")
        logger.info(f"未匹配: {unmatched_count} 条记录")
        return result_dict

    match_results = query_hospitals(site_dict_df, translated_df, config.META_MATCH_JSON_FILE)
    translated_df_indexed = translated_df.set_index('original_id')
    exact_partial_data = []
    exact_partial_ids = set()
    for match_type in ['exact_matches', 'partial_matches']:
        for query, matches in match_results.get(match_type, {}).items():
            for match in matches:
                oid = match['original_id']
                if oid not in exact_partial_ids:
                    original_record = translated_df_indexed.loc[oid]
                    exact_partial_data.append({
                        'original_id': oid,
                        'affiliation': original_record['affiliation'],
                        'affiliation_cn': original_record['affiliation_cn'],
                        'state': original_record['state'],
                        'city': original_record['city'],
                        'matched_site': match['hospital_name'],
                        'matched_site_en': match['hospital_name_en'],
                        'matched_site_province': match['hospital_province'],
                        'matched_site_city': match['hospital_city'],
                        'confidence': match['match_type']
                    })
                    exact_partial_ids.add(oid)

    exact_partial_df = pd.DataFrame(exact_partial_data)
    exact_partial_df.to_parquet(config.META_PARTIAL_EXACT_MATCHED_FILE, index=False)
    logger.info(f"精确/部分匹配找到 {len(exact_partial_df)} 条记录")

    ai_meta_input_data = []
    fuzzy_ids = {m['original_id'] for matches in match_results.get('fuzzy_matches', {}).values() for m in matches}
    fuzzy_ids -= exact_partial_ids
    for oid in fuzzy_ids:
        original_record = translated_df_indexed.loc[oid]
        query = original_record['affiliation_cn']
        matches = match_results.get('fuzzy_matches', {}).get(query, [])
        candidates = []
        seen_hospitals = set()
        for match in matches:
            if match['original_id'] == oid and match['hospital_name'] not in seen_hospitals:
                alias_list = match['hospital_alias']
                if isinstance(alias_list, np.ndarray):
                    alias_list = alias_list.tolist()
                elif not isinstance(alias_list, list):
                    alias_list = [str(alias_list)] if pd.notna(alias_list) else []
                candidates.append({
                    "hospital_name": match['hospital_name'],
                    "hospital_name_en": match['hospital_name_en'],
                    "hospital_alias": alias_list,
                    "hospital_province": match['hospital_province'],
                    "hospital_city": match['hospital_city']
                })
                seen_hospitals.add(match['hospital_name'])
        if candidates:
            ai_meta_input_data.append({
                'original_id': oid,
                'affiliation': original_record['affiliation'],
                'affiliation_cn': query,
                'state': original_record['state'],
                'city': original_record['city'],
                'candidates': candidates
            })

    if ai_meta_input_data:
        ai_meta_input_df = pd.DataFrame(ai_meta_input_data)
        ai_meta_input_df.to_parquet(config.AI_META_INPUT_FILE, index=False)
        logger.info(f"生成 {len(ai_meta_input_df)} 条模糊匹配候选")
    else:
        pd.DataFrame(columns=['original_id', 'affiliation', 'affiliation_cn', 'state', 'city', 'candidates']).to_parquet(config.AI_META_INPUT_FILE, index=False)
        logger.info("没有需要AI处理的模糊匹配记录。")

    logger.info("STEP 3 完成。")
    return True

if __name__ == "__main__":
    config = Config()
    step_3_candidate_matching(config)