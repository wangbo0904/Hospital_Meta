import streamlit as st
import pandas as pd
import time
import os
from collections import OrderedDict

# 导入您现有的脚本和配置
from config_utils import Config, BATCH_AI_SELECT_PROMPT
from step_1_english_match import step_1_initial_english_match
from step_2_translate import step_2_translate_unmatched
from step_3_candidate_matching import step_3_candidate_matching
from step_4_ai_candidate_selection import step_4_ai_candidate_selection
from step_5_ai_judgment import step_5_ai_judgment
from step_6_generate_final_report import generate_comprehensive_report

# --- 页面配置 ---
st.set_page_config(
    page_title="PI Site智能匹配AI流水线",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 自定义UI样式，实现“高级感”深色主题 ---
st.markdown("""
<style>
    /* 1. 全局与背景设置 */
    .stApp {
        background-color: #111827; /* 主背景：极深炭灰 */
        color: #D1D5DB; /* 默认文字：柔和浅灰 */
    }
    
    /* 2. 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #1F2937; /* 侧边栏背景：石墨灰 */
        border-right: 1px solid rgba(255, 255, 255, 0.1); /* 右侧辉光边框 */
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F9FAFB; /* 侧边栏标题：亮灰白 */
    }
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3,
    [data-testid="stSidebar"] label {
        color: #D1D5DB; /* 侧边栏标签与文字 */
    }
    [data-testid="stSidebarNavCollapseButton"] svg {
        fill: #D1D5DB; /* 折叠按钮图标 */
    }

    /* 3. 主内容区样式 */
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        color: #FFFFFF; /* 主内容区标题：纯白，更突出 */
    }

    /* 4. 动态与立体效果 */
    /* 容器的悬浮效果 */
    .st-emotion-cache-4oy321 {
        background-color: #1F2937; /* 容器背景：石墨灰 */
        border: 1px solid rgba(255, 255, 255, 0.1); /* 辉光边框 */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); /* 更柔和的阴影 */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .st-emotion-cache-4oy321:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4); /* 悬停时阴影加深 */
    }

    /* 按钮的立体效果 */
    .stButton>button {
        background-image: linear-gradient(to right, #3B82F6, #2563EB); /* 按钮背景：蓝色渐变 */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.75em 1.5em;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4); /* 悬停时带颜色的辉光阴影 */
    }
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* 输入控件的焦点效果 */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] {
        background-color: #374151; /* 输入框背景色 */
        border-radius: 6px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stSidebar"] input:focus,
    [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"]:focus-within {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5); /* 蓝色辉光 */
    }
</style>
""", unsafe_allow_html=True)

# --- 流水线步骤定义 (精简后) ---
PIPELINE_STEPS = OrderedDict([
    ("步骤 1: 英文名初步匹配", {"function": step_1_initial_english_match, "dependencies": []}),
    ("步骤 2: 翻译未匹配项", {"function": step_2_translate_unmatched, "dependencies": ["步骤 1: 英文名初步匹配"]}),
    ("步骤 3: 候选词匹配", {"function": step_3_candidate_matching, "dependencies": ["步骤 2: 翻译未匹配项"]}),
    ("步骤 4: AI筛选候选", {"function": step_4_ai_candidate_selection, "dependencies": ["步骤 3: 候选词匹配"]}),
    ("步骤 5: AI判断不一致项", {"function": step_5_ai_judgment, "dependencies": ["步骤 4: AI筛选候选"]}),
    ("步骤 6: 生成最终报告", {"function": generate_comprehensive_report, "dependencies": ["步骤 4: AI筛选候选", "步骤 5: AI判断不一致项"]})
])
STEP_NAMES = list(PIPELINE_STEPS.keys())

def get_execution_plan(selected_steps):
    """计算需要运行的完整步骤列表，包括所有依赖项。"""
    execution_plan = set()
    for step in selected_steps:
        execution_plan.add(step)
        dependencies_to_check = list(PIPELINE_STEPS[step]["dependencies"])
        while dependencies_to_check:
            dep = dependencies_to_check.pop(0)
            if dep not in execution_plan:
                execution_plan.add(dep)
                dependencies_to_check.extend(PIPELINE_STEPS[dep]["dependencies"])
    return [s for s in STEP_NAMES if s in execution_plan]

# --- UI: 左侧配置侧边栏 ---
with st.sidebar:
    st.image("https://staticcdn.pharmcube.com/images/activity/logo.png")
    st.title("PI Site智能匹配")
    st.info("ℹ️ 您可以使用顶部的'>'图标折叠此侧边栏。")
    
    st.header("1. 文件与运行模式")
    raw_data_file = st.file_uploader("上传原始数据文件", type=['parquet'])
    dict_file = st.file_uploader("上传机构词典文件", type=['parquet'])
    
    # --- 新增：清理结果的复选框 ---
    clear_old_results = st.checkbox("清理旧的运行结果 (建议新任务使用)", value=True)

    st.header("2. 使用者信息")
    organization = st.text_input("组织名称", value="DefaultOrg")

    st.header("3. 模型选择")
    translate_model = st.selectbox("翻译模型", ["gemini-2.5-flash-lite-nothinking", "gemini-2.5-flash-lite-preview-06-17-nothinking", "gemini-2.5-flash-lite-thinking"])
    ai_select_model = st.selectbox("AI筛选模型", ["gemini-2.5-flash-lite-nothinking", "gemini-2.5-pro-c", "gemini-2.5-pro-c-thinking", "gemini-2.5-flash-lite-preview-06-17-nothinking", "gemini-2.5-flash-lite-thinking"])
    ai_judge_model = st.selectbox("AI判断模型", ["gemini-2.5-flash-lite-nothinking", "gemini-2.5-pro-c", "gemini-2.5-pro-c-thinking", "gemini-2.5-flash-lite-preview-06-17-nothinking", "gemini-2.5-flash-lite-thinking"])

    st.header("4. 批次大小设置")
    ai_select_batch_size = st.slider("AI筛选批次 (步骤 4)", min_value=1, max_value=100, value=10)
    ai_judge_batch_size = st.slider("AI判断批次 (步骤 5)", min_value=1, max_value=100, value=10)

    st.header("5. 自定义提示")
    with st.expander("编辑AI候选筛选提示词"):
        custom_ai_select_prompt = st.text_area("提示词内容", value=BATCH_AI_SELECT_PROMPT, height=300)

# --- 主应用界面 ---
st.title("🚀 端到端机构名称匹配流水线")
st.markdown("请在左侧边栏配置参数、上传文件、选择要运行的步骤，然后启动流水线。")

with st.container(border=True):
    st.subheader("🛠️ 选择要运行的流水线步骤")
    st.markdown("选择您想执行的最终步骤。应用将自动运行所有必需的前置步骤。")
    
    selected_steps = st.multiselect("选择步骤", options=STEP_NAMES, default=[STEP_NAMES[-1]], label_visibility="collapsed")

if st.button("启动处理流水线", type="primary"):
    if not raw_data_file or not dict_file:
        st.error("❌ 请在开始前上传原始数据和机构词典两个文件。")
    elif not selected_steps:
        st.warning("⚠️ 请至少选择一个要运行的流水线步骤。")
    else:
        execution_plan = get_execution_plan(selected_steps)
        if set(execution_plan) != set(selected_steps):
            st.info(f"ℹ️ 为运行您的选择，将执行以下完整计划：\n" + " -> ".join([f"**{step.split(':')[0]}**" for step in execution_plan]))

        config = Config()
        
        # --- 根据复选框决定是否清理结果 ---
        if clear_old_results:
            st.info("ℹ️ 正在清理旧的运行结果...")
            config.clear_results_dir()
        else:
            st.warning("⚠️ 未清理旧结果。将尝试使用现有文件并跳过已完成的任务。")
            
        raw_data_path = config.DATA_DIR / raw_data_file.name
        dict_path = config.DATA_DIR / dict_file.name
        with open(raw_data_path, "wb") as f: f.write(raw_data_file.getbuffer())
        with open(dict_path, "wb") as f: f.write(dict_file.getbuffer())

        config.update_from_ui({
            "organization": organization, "translate_model": translate_model,
            "ai_select_model": ai_select_model, "ai_judge_model": ai_judge_model,
            "raw_data_file": {"name": raw_data_file.name}, "dict_file": {"name": dict_file.name}
        })
        
        config.AI_SELECT_BATCH_SIZE = ai_select_batch_size
        config.AI_JUDGE_BATCH_SIZE = ai_judge_batch_size
        globals()['BATCH_AI_SELECT_PROMPT'] = custom_ai_select_prompt
        
        overall_success = True
        with st.status("🚀 正在启动AI流水线...", expanded=True) as status:
            total_steps = len(execution_plan)
            for i, step_name in enumerate(execution_plan):
                status.update(label=f"正在执行 ({i+1}/{total_steps}): {step_name}...")
                step_function = PIPELINE_STEPS[step_name]["function"]
                try:
                    success = step_function(config)
                    if not success:
                        st.error(f"❌ {step_name} 失败。流水线已中止。")
                        status.update(label=f"流水线在 {step_name} 处失败", state="error")
                        overall_success = False
                        break
                    time.sleep(1)
                except Exception as e:
                    st.error(f"💥 在执行 {step_name} 期间发生意外错误: {e}")
                    status.update(label=f"流水线在 {step_name} 处崩溃", state="error")
                    overall_success = False
                    break
            
            if overall_success:
                status.update(label="✅ 流水线成功完成！", state="complete")

        if overall_success:
            st.success("🎉 所有选定步骤均已成功完成！")
            
            if "步骤 6: 生成最终报告" in execution_plan:
                report_path = config.FINAL_REPORT_OUTPUT_FILE
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    st.subheader("📊 最终综合报告")
                    st.components.v1.html(html_content, height=800, scrolling=True)

                    st.download_button(
                        label="📥 下载报告",
                        data=html_content,
                        file_name="final_comprehensive_report.html",
                        mime="text/html",
                    )
                else:
                    st.warning("未能找到最终报告文件，尽管该步骤已标记为成功。")