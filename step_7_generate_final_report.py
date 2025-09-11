#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_7_generate_final_report.py
@Description:
    独立的最终综合报告生成脚本。
    整合 step4, step5, step6 的结果，创建一个包含完整决策链的交互式HTML报告。
"""

import pandas as pd
import os
import datetime
from jinja2 import Environment, FileSystemLoader
import numpy as np
from tqdm import tqdm
# ==============================================================================
# 1. 配置
# ==============================================================================
class ReportConfig:
    RESULTS_DIR = "./SITE_APP/reports/"
    ALL_MATCHES_FILE = os.path.join(RESULTS_DIR, "step4_all_matches_combined.parquet")
    JUDGE_RESULTS_FILE = os.path.join(RESULTS_DIR, "step5_final_judge_output.parquet")
    ARBITRATION_FILE = os.path.join(RESULTS_DIR, "step6_arbitration_output.parquet")
    
    TEMPLATE_FILE = "./final_report_template.html"
    OUTPUT_REPORT_FILE = os.path.join(RESULTS_DIR, "final_comprehensive_report.html")

# ==============================================================================
# 2. 主执行函数
# ==============================================================================
def generate_comprehensive_report(config: ReportConfig):
    """
    【最终完整版】
    整合所有结果，并生成功能完备的交互式HTML报告。
    """
    print(">>> 开始生成最终综合报告...")

    # --- 1. 加载所有结果文件 ---
    try:
        df4 = pd.read_parquet(config.ALL_MATCHES_FILE)
        df5 = pd.read_parquet(config.JUDGE_RESULTS_FILE)
        df6 = pd.read_parquet(config.ARBITRATION_FILE)
        print(f"成功加载 Step 4({len(df4)}), Step 5({len(df5)}), Step 6({len(df6)}) 的结果。")
    except Exception as e:
        print(f"\033[91m错误：加载结果文件失败: {e}\033[0m")
        print("请确保您已完整运行所有前置步骤 (1-6)。")
        return

    # --- 2. 数据整合 ---
    print("正在整合所有决策数据...")
    # 强制统一ID类型为字符串，避免合并失败
    df4['original_id'] = df4['original_id'].astype(str)
    df5['original_id'] = df5['original_id'].astype(str)
    df6['original_id'] = df6['original_id'].astype(str)

    report_df = pd.merge(df4, df5[['original_id', 'is_same', 'reason']], on='original_id', how='left', suffixes=('_step4', '_step5'))
    report_df = pd.merge(report_df, df6[['original_id', 'decision', 'final_site', 'decision_reason']], on='original_id', how='left')
    
    # 统一填充空值
    report_df = report_df.fillna('')
    print(f"数据整合后总行数: {len(report_df)}")

    # --- 3. 数据分类 ---
    print("正在对数据进行分类...")
    CATEGORY_MAP = {
        'exact': '标签与匹配一致',
        'judged_consistent': 'AI判断: 一致',
        'arbitrated_matched': '最终裁决: Matched Site',
        'arbitrated_pi': '最终裁决: PI Site',
        'arbitrated_neither': '最终裁决: 皆不匹配',
        'pending': '未解决(待仲裁)',
        'unmatched': '未匹配'
    }

    def categorize_match(row):
        if row.get('decision'):
            if row['decision'] == 'pi_site_name': return 'arbitrated_pi'
            if row['decision'] == 'matched_site': return 'arbitrated_matched'
            return 'arbitrated_neither'
        if str(row.get('is_same')).lower() == 'true': return 'judged_consistent'
        if row.get('pi_site_name') == row.get('matched_site') and row.get('pi_site_name'): return 'exact'
        if not row.get('matched_site'): return 'unmatched'
        # is_same 为 False 但没有 decision 的情况
        if str(row.get('is_same')).lower() == 'false': return 'pending'
        return 'pending' # 其他所有待定情况

    report_df['category_key'] = report_df.apply(categorize_match, axis=1)
    report_df['category_display'] = report_df['category_key'].map(CATEGORY_MAP)

    # 确定 final_name
    def determine_final_name(row):
        if row['category_key'] in ['exact', 'judged_consistent', 'arbitrated_matched']:
            return row['matched_site']
        if row['category_key'] == 'arbitrated_pi':
            return row['pi_site_name']
        return '' # 其他情况最终名称为空
    
    report_df['final_name'] = report_df.apply(determine_final_name, axis=1)

    # --- 4. 计算统计数据 ---
    counts = report_df['category_display'].value_counts().to_dict()
    ordered_counts = {display_name: counts.get(display_name, 0) for key, display_name in CATEGORY_MAP.items()}
    print("统计数据计算完成。")

    # --- 5. 格式化HTML表格 ---
    print("正在格式化HTML表格...")
    html_rows = []
    for row in tqdm(report_df.to_dict('records'), desc="格式化HTML"):
        try:
            category_key = row.get('category_key', 'unmatched')
            
            reason_html = ""
            if row.get('reason_step4'):
                reason_html += f"<div class='reason-box reason-step4'><strong>Step 4 (AI选择理由):</strong> {row['reason_step4']}</div>"
            if row.get('reason_step5'):
                is_same_text = "✅ 一致" if str(row.get('is_same')).lower() == 'true' else "❌ 不一致"
                reason_html += f"<div class='reason-box reason-step5'><strong>Step 5 (判断理由):</strong> {is_same_text} - {row['reason_step5']}</div>"
            if row.get('decision_reason'):
                reason_html += f"<div class='reason-box reason-step6'><strong>Step 6 (仲裁理由):</strong> {row['decision_reason']}</div>"
            
            final_name = row.get('final_name', '')
            final_category = row.get('category_display', '')
            final_decision_html = f"<b>最终名称:</b> {final_name}" if final_name else "<b>最终名称:</b> <i style='color:grey;'>N/A</i>"
            
            decision_class = category_key
            final_decision_html += f"<div class='final-decision {decision_class}'>{final_category}</div>"

            html_rows.append(f"""
            <tr data-category="{category_key}">
                <td>{row.get('original_id', '')}</td>
                <td>
                    <div class="input-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                    <div class="input-block" style="margin-top: 10px;"><strong>PI Site Name:</strong><div>{row.get('pi_site_name', '')}</div></div>
                </td>
                <td>
                    <div class="input-block"><strong>匹配候选 (Matched Site):</strong><div>{row.get('matched_site', 'N/A')}</div></div>
                    {reason_html}
                </td>
                <td>{final_decision_html}</td>
            </tr>
            """)
        except Exception as e:
            print(f"\n\033[91m错误：格式化ID为 {row.get('original_id')} 的数据时失败: {e}\033[0m")

    table_body_html = "\n".join(html_rows)
    
    if not table_body_html:
        print("\n\033[91m严重错误：未能生成任何HTML表格行。报告将为空。\033[0m")

    # --- 6. 渲染并保存 ---
    print("正在渲染并保存HTML报告...")
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    template = env.get_template(config.TEMPLATE_FILE)
    
    html_content = template.render(
        source_file=f"{os.path.basename(config.ALL_MATCHES_FILE)}, {os.path.basename(config.JUDGE_RESULTS_FILE)}, {os.path.basename(config.ARBITRATION_FILE)}",
        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        category_map=CATEGORY_MAP,
        counts=ordered_counts,
        total_rows=len(report_df),
        table_body_html=table_body_html
    )

    with open(config.OUTPUT_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("\n" + "="*60)
    print(f"\033[92m>>> 最终综合报告生成成功！ <<<\033[0m")
    print(f"请在浏览器中打开文件: {os.path.abspath(config.OUTPUT_REPORT_FILE)}")
    print("="*60)
    
    return True

# ==============================================================================
# 3. 脚本执行入口
# ==============================================================================
if __name__ == "__main__":
    config = ReportConfig()
    generate_comprehensive_report(config)