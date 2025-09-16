#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: PI_SITE
@File: step_6_generate_final_report.py
@Description:
    独立的最终综合报告生成脚本。
    整合 step0, step4, step5的结果，创建包含完整决策链的交互式HTML报告。
    有Step 5和没有Step 5时使用完全不同的报告模板和逻辑，包含abnormal_data分类。
"""

import pandas as pd
import os
import sys
import datetime
from jinja2 import Environment, FileSystemLoader
import numpy as np
from tqdm import tqdm
from config_utils import Config, logger

def generate_report_with_step5(config, df0, df4, df5):
    """有Step 5结果时的报告生成逻辑，包含abnormal_data"""
    logger.info("使用有Step 5结果的报告模板，包含异常数据...")

    # 数据整合
    df4['esid'] = df4['esid'].astype(str)
    df5['esid'] = df5['esid'].astype(str)
    df0['esid'] = df0['esid'].astype(str)

    # 处理异常数据
    df0_abnormal = df0[df0['category'] != 'Valid Institution'].copy()
    df0_abnormal['category_key'] = 'abnormal'
    df0_abnormal['category_display'] = '异常数据'
    df0_abnormal['final_name'] = ''
    df0_abnormal['matched_site'] = ''
    df0_abnormal['reason_step4'] = ''
    df0_abnormal['reason_step5'] = ''
    df0_abnormal['is_same'] = pd.NA
    df0_abnormal['confidence'] = pd.NA
    df0_abnormal['trans_confidence'] = pd.NA
    df0_abnormal['affiliation_cn'] = ''
    
    # 合并 Step 4 和 Step 5 数据
    report_df = pd.merge(df4, df5[['esid', 'is_same', 'reason']], on='esid', how='left', suffixes=('_step4', '_step5'))

    # 处理 is_same 列
    def convert_is_same(val):
        if pd.isna(val):
            return pd.NA
        if val in [True, 'True', 'true', 1, '1']:
            return True
        if val in [False, 'False', 'false', 0, '0']:
            return False
        return pd.NA
    
    report_df['is_same'] = report_df['is_same'].apply(convert_is_same).astype('boolean')
    report_df = report_df.fillna({'reason_step5': '', 'confidence': np.nan})
    
    # 数据分类 - 有Step 5的完整分类
    CATEGORY_MAP = {
        'exact': '标签与匹配一致',
        'judged_consistent': 'AI判断: 一致',
        'judged_inconsistent': 'AI判断: 不一致（需人工核查）',
        'unmatched': '未匹配',
        'abnormal': '异常数据'
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
    report_df['final_name'] = report_df.apply(
        lambda row: row['matched_site'] if row['category_key'] in ['exact', 'judged_consistent'] else '', axis=1
    )

    # 合并异常数据
    report_df = pd.concat([report_df, df0_abnormal], ignore_index=True, sort=False)
    
    # 获取唯一的置信度值用于筛选（直接使用原始值）
    def get_confidence_options(column_name):
        values = report_df[column_name].dropna().unique()
        options = ['all'] + [str(v) for v in sorted(values)]
        return options

    confidence_options = get_confidence_options('confidence')
    trans_confidence_options = get_confidence_options('trans_confidence')

    # 格式化HTML表格
    html_rows = []
    for row in tqdm(report_df.to_dict('records'), desc="格式化HTML(有Step5)"):
        try:
            category_key = row.get('category_key', 'unmatched')
            
            if category_key == 'abnormal':
                # 异常数据专用格式
                reason_html = f"<div class='reason-box reason-abnormal'><strong>异常原因:</strong> {row.get('reason', 'N/A')}</div>"
                final_decision_html = f"<div class='final-decision abnormal'>异常数据 ({row.get('category', 'N/A')})</div>"
                
                html_rows.append(f"""
                <tr data-category="{category_key}" data-confidence="N/A" data-trans-confidence="N/A">
                    <td>{row.get('esid', '')}</td>
                    <td>
                        <div class="input-block affiliation-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                    </td>
                    <td><div class='trans-confidence-block'>N/A</div></td>
                    <td>
                        <div class="input-block"><strong>匹配候选 (Matched Site):</strong><div>{row.get('matched_site', 'N/A')}</div></div>
                        {reason_html}
                    </td>
                    <td><div class='confidence-block'>N/A</div></td>
                    <td>{final_decision_html}</td>
                </tr>
                """)
            else:
                # 正常数据格式
                reason_html = ""
                if row.get('reason_step4'):
                    reason_html += f"<div class='reason-box reason-step4'><strong>Step 4 (AI选择理由):</strong> {row['reason_step4']}</div>"
                if row.get('reason_step5'):
                    is_same_text = "✅ 一致" if pd.notna(row.get('is_same')) and bool(row['is_same']) else "❌ 不一致"
                    reason_html += f"<div class='reason-box reason-step5'><strong>Step 5 (判断理由):</strong> {is_same_text} - {row['reason_step5']}</div>"
                
                final_name = row.get('final_name', '')
                final_category = row.get('category_display', '')
                final_decision_html = f"<b>最终名称:</b> {final_name}" if final_name else "<b>最终名称:</b> <i style='color:grey;'>N/A</i>"
                final_decision_html += f"<div class='final-decision {category_key}'>{final_category}</div>"

                confidence_value = row.get('confidence', '')
                confidence_str = str(confidence_value) if pd.notna(confidence_value) and confidence_value != '' else "N/A"
                confidence_display = f"<div class='confidence-block'>{confidence_str}</div>"

                trans_confidence_value = row.get('trans_confidence', '')
                trans_confidence_str = str(trans_confidence_value) if pd.notna(trans_confidence_value) and trans_confidence_value != '' else "N/A"
                trans_confidence_display = f"<div class='trans-confidence-block'>{trans_confidence_str}</div>"

                html_rows.append(f"""
                <tr data-category="{category_key}" data-confidence="{confidence_str}" data-trans-confidence="{trans_confidence_str}">
                    <td>{row.get('esid', '')}</td>
                    <td>
                        <div class="input-block affiliation-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                        <div class="input-block pi-site-block" style="margin-top: 10px;"><strong>PI Site Name:</strong><div>{row.get('pi_site_name', '')}</div></div>
                    </td>
                    <td>{trans_confidence_display}</td>
                    <td>
                        <div class="input-block"><strong>匹配候选 (Matched Site):</strong><div>{row.get('matched_site', 'N/A')}</div></div>
                        {reason_html}
                    </td>
                    <td>{confidence_display}</td>
                    <td>{final_decision_html}</td>
                </tr>
                """)
        except Exception as e:
            logger.error(f"格式化ID为 {row.get('esid')} 的数据时失败: {e}")

    return report_df, CATEGORY_MAP, html_rows, confidence_options, trans_confidence_options

def generate_report_without_step5(config, df0, df4):
    """没有Step 5结果时的报告生成逻辑 - 增量数据报告，包含abnormal_data"""
    logger.info("使用增量数据报告模板，包含异常数据...")
    
    # 处理异常数据
    df0_abnormal = df0[df0['category'] != 'Valid Institution'].copy()
    df0_abnormal['category_key'] = 'abnormal'
    df0_abnormal['category_display'] = '异常数据'
    df0_abnormal['final_name'] = ''
    df0_abnormal['matched_site'] = ''
    df0_abnormal['reason'] = df0_abnormal['reason'].fillna('N/A')
    df0_abnormal['confidence'] = pd.NA
    df0_abnormal['trans_confidence'] = pd.NA
    df0_abnormal['affiliation_cn'] = ''
    
    # 处理 Step 4 数据
    report_df = df4.copy()
    
    # 数据分类 - 增量数据的分类
    CATEGORY_MAP = {
        'matched': '有匹配结果',
        'unmatched': '未匹配',
        'abnormal': '异常数据'
    }

    def categorize_match(row):
        if pd.notna(row.get('matched_site')) and row.get('matched_site') != '':
            return 'matched'
        return 'unmatched'

    report_df['category_key'] = report_df.apply(categorize_match, axis=1)
    report_df['category_display'] = report_df['category_key'].map(CATEGORY_MAP)
    report_df['final_name'] = report_df['matched_site']

    # 合并异常数据
    report_df = pd.concat([report_df, df0_abnormal], ignore_index=True, sort=False)
    
    # 获取唯一的置信度值用于筛选（直接使用原始值）
    def get_confidence_options(column_name):
        values = report_df[column_name].dropna().unique()
        options = ['所有置信度'] + [str(v) for v in sorted(values)]
        return options

    confidence_options = get_confidence_options('confidence')
    trans_confidence_options = get_confidence_options('trans_confidence')

    # 格式化HTML表格
    html_rows = []
    for row in tqdm(report_df.to_dict('records'), desc="格式化HTML(增量数据)"):
        try:
            category_key = row.get('category_key', 'unmatched')
            
            if category_key == 'abnormal':
                # 异常数据专用格式
                reason_html = f"<div class='reason-box reason-abnormal'><strong>异常原因:</strong> {row.get('reason', 'N/A')}</div>"
                final_decision_html = f"<div class='final-decision abnormal'>异常数据 ({row.get('category', 'N/A')})</div>"
                html_rows.append(f"""
                <tr data-category="{category_key}" data-confidence="N/A" data-trans-confidence="N/A">
                    <td>{row.get('esid', '')}</td>
                    <td>
                        <div class="input-block affiliation-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                    </td>
                    <td>
                        <div class='trans-confidence-block' data-confidence="N/A">N/A</div>
                    </td>
                    <td>{reason_html}</td>
                    <td>
                        <div class='confidence-block' data-confidence="N/A">N/A</div>
                    </td>
                    <td>{final_decision_html}</td>
                </tr>
                """)
            else:
                # 原有格式
                reason_html = ""
                if row.get('reason'):
                    reason_html += f"<div class='reason-box reason-match'><strong>匹配原因:</strong> {row['reason']}</div>"
                
                final_name = row.get('final_name', '')
                final_category = row.get('category_display', '')
                
                if category_key == 'matched':
                    final_decision_html = f"<b>匹配结果:</b> {final_name}"
                else:
                    final_decision_html = f"<b>匹配结果:</b> <i style='color:#666;'>未匹配到机构</i>"
                
                final_decision_html += f"<div class='final-decision {category_key}'>{final_category}</div>"

                confidence_value = row.get('confidence', '')
                confidence_display = str(confidence_value) if pd.notna(confidence_value) and confidence_value != '' else "N/A"

                trans_confidence_value = row.get('trans_confidence', '')
                trans_display = str(trans_confidence_value) if pd.notna(trans_confidence_value) and trans_confidence_value != '' else "N/A"

                html_rows.append(f"""
                <tr data-category="{category_key}" data-confidence="{confidence_display}" data-trans-confidence="{trans_display}">
                    <td>{row.get('esid', '')}</td>
                    <td>
                        <div class="input-block affiliation-block"><strong>Affiliation:</strong><div>{row.get('affiliation', '')}</div></div>
                        <div class="input-block affiliation-cn-block" style="margin-top: 10px;"><strong>Affiliation CN:</strong><div>{row.get('affiliation_cn', '')}</div></div>
                    </td>
                    <td>
                        <div class='trans-confidence-block' data-confidence="{trans_display}">{trans_display}</div>
                    </td>
                    <td>
                        <div class="input-block"><strong>匹配候选:</strong><div>{row.get('matched_site', 'N/A')}</div></div>
                        {reason_html}
                    </td>
                    <td>
                        <div class='confidence-block' data-confidence="{confidence_display}">{confidence_display}</div>
                    </td>
                    <td>{final_decision_html}</td>
                </tr>
                """)
        except Exception as e:
            logger.error(f"格式化ID为 {row.get('esid')} 的数据时失败: {e}")

    return report_df, CATEGORY_MAP, html_rows, confidence_options, trans_confidence_options

def generate_comprehensive_report(config):
    """
    整合所有结果，并生成功能完备的交互式HTML报告，同时导出最终结果为Parquet文件。
    有Step 5和没有Step 5时使用完全不同的逻辑，包含abnormal_data分类。
    """
    logger.info("\n" + "="*50)
    logger.info("STEP 7: 正在生成最终综合报告...")
    logger.info("="*50)

    # --- 1. 加载所有结果文件 ---
    try:
        df4 = pd.read_parquet(config.ALL_MATCHES_COMBINED_FILE)
        logger.info(f"成功加载 Step 4 的结果 ({len(df4)} 行)。")
    except Exception as e:
        logger.error(f"加载 Step 4 结果文件失败: {e}")
        logger.error("请确保您已运行 Step 4。")
        return False

    # 加载异常数据
    try:
        df0 = pd.read_parquet(config.ABNORMAL_DATA_FILE)
        logger.info(f"成功加载 Step 0 的异常数据 ({len(df0)} 行)。")
    except Exception as e:
        logger.warning(f"加载 Step 0 异常数据文件失败，将忽略异常数据: {e}")
        df0 = pd.DataFrame(columns=['esid', 'affiliation', 'category', 'reason'])

    # 检查Step 5结果是否存在
    has_step5 = os.path.exists(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE)
    df5 = None
    
    if has_step5:
        try:
            df5 = pd.read_parquet(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE)
            logger.info(f"成功加载 Step 5 的结果 ({len(df5)} 行)。")
        except Exception as e:
            logger.warning(f"加载 Step 5 结果文件失败，将使用无Step 5的报告模式: {e}")
            has_step5 = False

    # --- 2. 选择不同的报告生成逻辑 ---
    if has_step5 and df5 is not None and not df5.empty:
        report_df, category_map, html_rows, confidence_options, trans_confidence_options = generate_report_with_step5(config, df0, df4, df5)
        template_file = config.FINAL_REPORT_TEMPLATE_FILE  # 有Step 5的模板
        report_type = "complete"
    else:
        report_df, category_map, html_rows, confidence_options, trans_confidence_options = generate_report_without_step5(config, df0, df4)
        template_file = config.FINAL_REPORT_TEMPLATE_NO_STEP5_FILE  # 无Step 5的模板
        report_type = "partial"
        logger.info("使用无Step 5的报告模式")

    # --- 3. 保存最终结果到 Parquet 文件 ---
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

    # --- 4. 计算统计数据 ---
    counts = report_df['category_display'].value_counts().to_dict()
    ordered_counts = {display_name: counts.get(display_name, 0) for key, display_name in category_map.items()}
    logger.info("统计数据计算完成。")

    # --- 5. 渲染并保存 HTML 报告 ---
    logger.info("正在渲染并保存HTML报告...")
    template_dir = os.path.dirname(template_file)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(os.path.basename(template_file))

    # 添加数据源信息
    source_files = [os.path.basename(config.ALL_MATCHES_COMBINED_FILE), os.path.basename(config.ABNORMAL_DATA_FILE)]
    if has_step5 and df5 is not None:
        source_files.append(os.path.basename(config.FINAL_JUDGE_OUTPUT_PARQUET_FILE))

    html_content = template.render(
        source_file=", ".join(source_files),
        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        category_map=category_map,
        counts=ordered_counts,
        total_rows=len(report_df),
        table_body_html="\n".join(html_rows),
        has_step5_results=has_step5 and df5 is not None,
        confidence_options=confidence_options,
        trans_confidence_options=trans_confidence_options
    )

    # 确定输出文件路径
    output_file = config.FINAL_REPORT_OUTPUT_FILE if has_step5 else config.FINAL_REPORT_TEMPLATE_NO_STEP5_FILE.replace('.html', '_output.html')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    logger.info("\n" + "="*60)
    logger.info(f">>> 最终综合报告生成成功！ <<<")
    logger.info(f"报告类型: {'完整报告(含Step 5)' if report_type == 'complete' else '部分报告(无Step 5)'}")
    logger.info(f"请在浏览器中打开文件: {os.path.abspath(output_file)}")
    logger.info(f"最终结果 Parquet 文件已保存到: {os.path.abspath(config.FINAL_OUTPUT)}")
    
    if report_type == "partial":
        logger.info(f"\033[93m注意: 此报告仅基于 Step 0 和 Step 4 的结果生成，Step 5 的结果未找到或加载失败。\033[0m")
    
    logger.info("="*60)
    
    return True
    
if __name__ == "__main__":
    print("正在尝试使用默认配置进行一次独立运行测试...")
    
    try:
        config = Config()
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        success = generate_comprehensive_report(config)
        if not success:
            print("报告生成失败，请检查前置步骤是否已完成。")
            sys.exit(1)
    except Exception as e:
        print(f"独立运行时发生错误: {e}")
        sys.exit(1)