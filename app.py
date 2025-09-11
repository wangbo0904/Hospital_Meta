# app.py
import streamlit as st
import pandas as pd
import os
import time
from threading import Thread
from datetime import datetime

# 导入所有需要的模块和函数
from config_utils import Config, logger, BATCH_TRANSLATE_PROMPT, BATCH_AI_SELECT_PROMPT, BATCH_JUDGE_PROMPT, BATCH_ARBITRATE_PROMPT
from step_1_english_match import step_1_initial_english_match
from step_2_translate import step_2_translate_unmatched
from step_3_candidate_matching import step_3_candidate_matching
from step_4_ai_candidate_selection import step_4_ai_candidate_selection
from step_5_ai_judgment import step_5_ai_judgment
from step_6_final_arbitration import step_6_ai_arbitration
from step_7_generate_final_report import generate_comprehensive_report

# --- 页面配置 ---
st.set_page_config(
    page_title="医疗机构名称匹配系统",
    page_icon="🏥",
    layout="wide",
)

# --- 会话状态管理 ---
if 'prompts' not in st.session_state:
    st.session_state.prompts = {
        "translate": BATCH_TRANSLATE_PROMPT,
        "select": BATCH_AI_SELECT_PROMPT,
        "judge": BATCH_JUDGE_PROMPT,
        "arbitrate": BATCH_ARBITRATE_PROMPT
    }
if 'config' not in st.session_state:
    st.session_state.config = Config()

# --- 侧边栏 ---
st.sidebar.title("导航")
page = st.sidebar.radio("选择一个页面", ["🏠 主页 & 配置", "📝 Prompt 编辑器", "🚀 执行 & 结果"])

# ==============================================================================
# 页面一：主页 & 配置
# ==============================================================================
if page == "🏠 主页 & 配置":
    st.title("🏥 医疗机构名称智能匹配与标准化系统")
    st.markdown("### 欢迎使用！")
    st.markdown("这是一个端到端的流水线，用于清洗、匹配、验证和仲裁医疗机构名称。")
    
    st.markdown("---")
    st.subheader("⚙️ 核心参数配置")
    
    cfg = st.session_state.config
    cfg.MAX_WORKERS = st.slider("最大并发数 (MAX_WORKERS)", 1, 50, cfg.MAX_WORKERS)
    cfg.CANDIDATE_LIMIT = st.slider("模糊匹配候选数量 (CANDIDATE_LIMIT)", 5, 50, cfg.CANDIDATE_LIMIT)
    
    st.subheader("📁 文件路径配置")
    cfg.RAW_PARQUET_FILE = st.text_input("原始数据文件路径", cfg.RAW_PARQUET_FILE)
    cfg.SITE_DICT_FILE = st.text_input("机构字典文件路径", cfg.SITE_DICT_FILE)
    cfg.RESULTS_DIR = st.text_input("结果输出目录", cfg.RESULTS_DIR)

    if st.button("保存配置"):
        st.session_state.config = cfg
        st.success("配置已保存！")

# ==============================================================================
# 页面二：Prompt 编辑器
# ==============================================================================
elif page == "📝 Prompt 编辑器":
    st.title("📝 Prompt 编辑器")
    st.info("在这里，您可以实时查看和修改用于AI处理的系统提示词 (Prompt)。修改将保存在当前会话中。")

    prompts = st.session_state.prompts
    
    with st.expander("Step 2: 翻译 Prompt", expanded=True):
        prompts['translate'] = st.text_area("Translate Prompt", prompts['translate'], height=400, key="p_trans")

    with st.expander("Step 4: AI选择 Prompt"):
        prompts['select'] = st.text_area("Select Prompt", prompts['select'], height=400, key="p_select")
    
    with st.expander("Step 5: AI判断 Prompt"):
        prompts['judge'] = st.text_area("Judge Prompt", prompts['judge'], height=400, key="p_judge")
        
    with st.expander("Step 6: AI仲裁 Prompt"):
        prompts['arbitrate'] = st.text_area("Arbitrate Prompt", prompts['arbitrate'], height=400, key="p_arb")

    st.session_state.prompts = prompts
    st.success("Prompt 已在当前会话中实时更新。")

# ==============================================================================
# 页面三：执行 & 结果
# ==============================================================================
elif page == "🚀 执行 & 结果":
    st.title("🚀 执行流水线 & 查看结果")

    # 创建一个动态的配置对象，它会使用 session_state 中最新的 prompts
    dynamic_config = st.session_state.config
    dynamic_config.BATCH_TRANSLATE_PROMPT = st.session_state.prompts['translate']
    dynamic_config.BATCH_AI_SELECT_PROMPT = st.session_state.prompts['select']
    dynamic_config.BATCH_JUDGE_PROMPT = st.session_state.prompts['judge']
    dynamic_config.BATCH_ARBITRATE_PROMPT = st.session_state.prompts['arbitrate']

    steps_to_run = st.multiselect(
        "选择要执行的步骤 (将按顺序运行):",
        options=[
            "Step 1: 英文精确匹配",
            "Step 2: AI 翻译",
            "Step 3: 候选匹配",
            "Step 4: AI 候选选择",
            "Step 5: AI 判断",
            "Step 6: AI 最终仲裁",
            "Step 7: 生成最终报告"
        ],
        default=[
            "Step 1: 英文精确匹配",
            "Step 2: AI 翻译",
            "Step 3: 候选匹配",
            "Step 4: AI 候选选择",
            "Step 5: AI 判断",
            "Step 6: AI 最终仲裁",
            "Step 7: 生成最终报告"
        ]
    )

    if st.button("🚀 执行所选步骤"):
        # 确保结果目录存在
        os.makedirs(dynamic_config.RESULTS_DIR, exist_ok=True)
        
        log_area = st.empty()
        log_messages = ["### 🚀 流水线执行日志\n\n"]
        
        def run_step(step_func, step_name):
            log_messages.append(f"**{datetime.now().strftime('%H:%M:%S')} - 正在执行 {step_name}...**\n")
            log_area.markdown("".join(log_messages))
            success = step_func(dynamic_config)
            if success:
                log_messages.append(f"**{datetime.now().strftime('%H:%M:%S')} - ✅ {step_name} 完成！**\n\n")
            else:
                log_messages.append(f"**{datetime.now().strftime('%H:%M:%S')} - ❌ {step_name} 失败！流水线终止。**\n\n")
            log_area.markdown("".join(log_messages))
            return success

        with st.spinner("正在处理..."):
            pipeline_steps = {
                "Step 1: 英文精确匹配": step_1_initial_english_match,
                "Step 2: AI 翻译": step_2_translate_unmatched,
                "Step 3: 候选匹配": step_3_candidate_matching,
                "Step 4: AI 候选选择": step_4_ai_candidate_selection,
                "Step 5: AI 判断": step_5_ai_judgment,
                "Step 6: AI 最终仲裁": step_6_ai_arbitration,
                "Step 7: 生成最终报告": generate_comprehensive_report
            }
            
            for step_name in steps_to_run:
                if not run_step(pipeline_steps[step_name], step_name):
                    break # 如果任何一步失败，则终止
        
        st.success("所选步骤执行完毕！")

    st.markdown("---")
    st.subheader("📊 查看最终报告")
    
    report_path = dynamic_config.OUTPUT_REPORT_FILE
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.info("尚未生成报告。请执行“Step 7: 生成最终报告”来查看结果。")