#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_6_generate_final_report.py
@Description:
    独立的最终综合报告生成脚本。
    整合 step4, step5的结果，创建一个包含完整决策链的交互式HTML报告。
"""

import pandas as pd
import os
import sys
import datetime
from jinja2 import Environment, FileSystemLoader
import numpy as np
from tqdm import tqdm
from config_utils import Config, logger

def generate_comprehensive_report(config):
    """
    整合所有结果，并生成功能完备的交互式HTML报告，同时导出最终结果为Parquet文件。
    """
    logger.info("\n" + "="*50)
    logger.info("STEP 7: 正在生成最终综合报告...")
    logger.info("="*50)

    # --- 1. 加载所有结果文件 ---
    try:
        df4 = pd.read_parquet(config.ALL_MATCHES_COMBINED_FILE)
        df5 = pd.read_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE)
        logger.info(f"成功加载 Step 4({len(df4)}), Step 5({len(df5)}) 的结果。")
    except Exception as e:
        logger.error(f"加载结果文件失败: {e}")
        logger.error("请确保您已完整运行所有前置步骤 (1-5)。")
        return False

    # --- 2. 数据整合 ---
    logger.info("正在整合所有决策数据...")
    # 强制统一ID类型为字符串，避免合并失败
    df4['original_id'] = df4['original_id'].astype(str)
    df5['original_id'] = df5['original_id'].astype(str)

    # 合并数据，包含 step4 的 confidence 列
    report_df = pd.merge(df4, df5[['original_id', 'is_same', 'reason']], on='original_id', how='left', suffixes=('_step4', '_step5'))
    
    # 处理 is_same 列：转换为 nullable boolean 类型
    def convert_is_same(val):
        if pd.isna(val):
            return pd.NA
        if val in [True, 'True', 'true', 1, '1']:
            return True
        if val in [False, 'False', 'false', 0, '0']:
            return False
        return pd.NA
    
    report_df['is_same'] = report_df['is_same'].apply(convert_is_same).astype('boolean')
    
    # 填充其他空值
    report_df = report_df.fillna({'reason': '', 'confidence': np.nan})
    logger.info(f"数据整合后总行数: {len(report_df)}")

    # --- 3. 数据分类 ---
    logger.info("正在对数据进行分类...")
    CATEGORY_MAP = {
        'exact': '标签与匹配一致',
        'judged_consistent': 'AI判断: 一致',
        'judged_inconsistent': 'AI判断: 不一致（需人工核查）',
        'unmatched': '未匹配'
    }

    def categorize_match(row):
        if row.get('pi_site_name') == row.get('matched_site') and row.get('pi_site_name'): 
            return 'exact'
        if pd.notna(row.get('is_same')) and bool(row['is_same']): 
            return 'judged_consistent'
        if pd.notna(row.get('is_same')) and not bool(row['is_same']): 
            return 'judged_inconsistent'
        if not row.get('matched_site'): 
            return 'unmatched'
        return 'unmatched'

    report_df['category_key'] = report_df.apply(categorize_match, axis=1)
    report_df['category_display'] = report_df['category_key'].map(CATEGORY_MAP)

    def determine_final_name(row):
        if row['category_key'] in ['exact', 'judged_consistent']:
            return row['matched_site']
        return ''
    
    report_df['final_name'] = report_df.apply(determine_final_name, axis=1)

    # --- 4. 保存最终结果到 Parquet 文件 ---
    logger.info("正在保存最终结果到 Parquet 文件...")
    output_dir = os.path.dirname(config.FINAL_OUTPUT)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        report_df.to_parquet(config.FINAL_OUTPUT, index=False)
        logger.info(f"最终结果已保存到: {config.FINAL_OUTPUT}")
    except Exception as e:
        logger.error(f"保存 Parquet 文件失败: {e}")
        return False

    # --- 5. 计算统计数据 ---
    counts = report_df['category_display'].value_counts().to_dict()
    ordered_counts = {display_name: counts.get(display_name, 0) for key, display_name in CATEGORY_MAP.items()}
    logger.info("统计数据计算完成。")

    # --- 6. 格式化HTML表格 ---
    print("正在格式化HTML表格...")
    html_rows = []
    for row in tqdm(report_df.to_dict('records'), desc="格式化HTML"):
        try:
            category_key = row.get('category_key', 'unmatched')
            
            reason_html = ""
            if row.get('reason_step4'):
                reason_html += f"<div class='reason-box reason-step4'><strong>Step 4 (AI选择理由):</strong> {row['reason_step4']}</div>"
            if row.get('reason_step5'):
                is_same_text = "✅ 一致" if pd.notna(row.get('is_same')) and bool(row['is_same']) else "❌ 不一致"
                reason_html += f"<div class='reason-box reason-step5'><strong>Step 5 (判断理由):</strong> {is_same_text} - {row['reason_step5']}</div>"
            
            final_name = row.get('final_name', '')
            final_category = row.get('category_display', '')
            final_decision_html = f"<b>最终名称:</b> {final_name}" if final_name else "<b>最终名称:</b> <i style='color:grey;'>N/A</i>"
            
            decision_class = category_key
            final_decision_html += f"<div class='final-decision {decision_class}'>{final_category}</div>"

            confidence_value = row.get('confidence', '')
            if pd.isna(confidence_value) or confidence_value == '':
                confidence_display = "<div class='confidence-block'>N/A</div>"
            else:
                try:
                    conf_float = float(confidence_value)
                    confidence_display = f"<div class='confidence-block'>{conf_float:.2%}</div>"
                except ValueError:
                    confidence_display = f"<div class='confidence-block'>{confidence_value}</div>"

            html_rows.append(f"""
            <tr data-category="{category_key}">
                <td>{row.get('original_id', '')}</td>
                <td>
                    <div class="input-block affiliation-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                    <div class="input-block pi-site-block" style="margin-top: 10px;"><strong>PI Site Name:</strong><div>{row.get('pi_site_name', '')}</div></div>
                </td>
                <td>
                    <div class="input-block"><strong>匹配候选 (Matched Site):</strong><div>{row.get('matched_site', 'N/A')}</div></div>
                    {reason_html}
                </td>
                <td>{confidence_display}</td>
                <td>{final_decision_html}</td>
            </tr>
            """)
        except Exception as e:
            print(f"\n\033[91m错误：格式化ID为 {row.get('original_id')} 的数据时失败: {e}\033[0m")

    table_body_html = "\n".join(html_rows)
    
    if not table_body_html:
        print("\n\033[91m严重错误：未能生成任何HTML表格行。报告将为空。\033[0m")

    # --- 7. 渲染并保存 HTML 报告 ---
    print("正在渲染并保存HTML报告...")
    template_dir = os.path.dirname(config.FINAL_REPORT_TEMPLATE_FILE)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(os.path.basename(config.FINAL_REPORT_TEMPLATE_FILE))

    html_content = template.render(
        source_file=f"{os.path.basename(config.ALL_MATCHES_COMBINED_FILE)}, {os.path.basename(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE)}",
        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        category_map=CATEGORY_MAP,
        counts=ordered_counts,
        total_rows=len(report_df),
        table_body_html=table_body_html
    )

    output_dir = os.path.dirname(config.FINAL_REPORT_OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(config.FINAL_REPORT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("\n" + "="*60)
    print(f"\033[92m>>> 最终综合报告生成成功！ <<<\033[0m")
    print(f"请在浏览器中打开文件: {os.path.abspath(config.FINAL_REPORT_OUTPUT_FILE)}")
    print(f"最终结果 Parquet 文件已保存到: {os.path.abspath(config.FINAL_OUTPUT)}")
    print("="*60)
    
    return True

# 如果这个文件被直接运行，可以提供一个测试入口
if __name__ == "__main__":
    print("正在尝试使用默认配置进行一次独立运行测试...")
    
    try:
        config = Config()
        # 确保结果目录存在
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        success = generate_comprehensive_report(config)
        if not success:
            print("报告生成失败，请检查前置步骤是否已完成。")
            sys.exit(1)
    except Exception as e:
        print(f"独立运行时发生错误: {e}")
        sys.exit(1)