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
# 会话状态管理 (核心修改)
# ==============================================================================
# 使用 st.session_state 来持久化用户的配置
if 'config' not in st.session_state:
    # 1. 尝试从 Streamlit secrets 加载密钥
    try:
        openai_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
        genai_key = st.secrets["api_keys"]["GENAI_API_KEY"]
    except:
        openai_key = "" # 如果 secrets 中没有，则为空
        genai_key = ""

    # 2. 初始化一个 Config 对象并存入 session_state
    #    这里的值将作为用户界面的默认值
    config = Config()
    config.OPENAI_API_KEY = openai_key
    config.GENAI_API_KEY = genai_key
    # 设置其他默认值
    config.OPENAI_BASE_URL = "http://116.63.133.80:30660/api/llm/v1"
    config.GENAI_BASE_URL = "https://globalai.vip/"
    config.API_PROJECT = "PI_SITE"
    config.ORGANIZATION = "WB"
    config.TRANSLATE_MODEL = "global-gemini-2.5-pro"
    config.AI_SELECT_MODEL = "gemini-2.5-flash-lite-nothinking"
    config.AI_JUDGE_MODEL = "gemini-2.5-flash-lite-nothinking"
    config.ARBITRATE_MODEL = "gemini-2.5-flash-lite-nothinking"
    # ... (其他非敏感配置)
    st.session_state.config = config

# ... (prompts, data_loaded, results_df 的 session_state 初始化保持不变)

# ==============================================================================
# 页面一：主页 & 配置 (全新版本)
# ==============================================================================
if page == "🏠 主页 & 配置":
    st.title("🏥 医疗机构名称智能匹配与标准化系统")
    st.markdown("欢迎使用！请在下方配置您的API信息和文件路径。")
    st.info("🔑 **安全提示**: 您的API密钥只会保存在当前浏览器会话中，不会被存储或上传。")

    # 从 session_state 中获取当前的配置对象
    cfg = st.session_state.config

    with st.expander("🔑 API 与模型配置", expanded=True):
        st.subheader("OpenAI-Compatible API (用于 Step 2)")
        cfg.OPENAI_BASE_URL = st.text_input("API Base URL", value=cfg.OPENAI_BASE_URL)
        cfg.OPENAI_API_KEY = st.text_input("API Key", value=cfg.OPENAI_API_KEY, type="password")
        cfg.TRANSLATE_MODEL = st.text_input("翻译模型名称", value=cfg.TRANSLATE_MODEL)

        st.markdown("---")
        st.subheader("Google GenAI API (用于 Step 4, 5, 6)")
        cfg.GENAI_BASE_URL = st.text_input("GenAI Base URL", value=cfg.GENAI_BASE_URL)
        cfg.GENAI_API_KEY = st.text_input("GenAI API Key", value=cfg.GENAI_API_KEY, type="password")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            cfg.AI_SELECT_MODEL = st.text_input("选择模型", value=cfg.AI_SELECT_MODEL)
        with col2:
            cfg.AI_JUDGE_MODEL = st.text_input("判断模型", value=cfg.AI_JUDGE_MODEL)
        with col3:
            cfg.ARBITRATE_MODEL = st.text_input("仲裁模型", value=cfg.ARBITRATE_MODEL)

    with st.expander("📁 文件与性能配置"):
        cfg.RAW_PARQUET_FILE = st.text_input("原始数据文件路径", value=cfg.RAW_PARQUET_FILE)
        cfg.SITE_DICT_FILE = st.text_input("机构字典文件路径", value=cfg.SITE_DICT_FILE)
        cfg.RESULTS_DIR = st.text_input("结果输出目录", value=cfg.RESULTS_DIR)
        cfg.MAX_WORKERS = st.slider("最大并发数", 1, 50, cfg.MAX_WORKERS)
        cfg.CANDIDATE_LIMIT = st.slider("模糊匹配候选数量", 5, 50, cfg.CANDIDATE_LIMIT)

    # 每次交互后，Streamlit会自动重新运行，配置会实时保存在 session_state 中
    st.session_state.config = cfg
    
    if st.button("✅ 确认配置"):
        st.success("配置已在当前会话中更新！")

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