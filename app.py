# app.py (Shiny for Python Version)
import pandas as pd
import os
import time
from datetime import datetime

# Shiny 核心库
from shiny import App, render, ui, reactive
import shinyswatch

# 导入所有需要的模块和函数
from config_utils import Config, logger, BATCH_TRANSLATE_PROMPT, BATCH_AI_SELECT_PROMPT, BATCH_JUDGE_PROMPT, BATCH_ARBITRATE_PROMPT
from step_1_english_match import step_1_initial_english_match
from step_2_translate import step_2_translate_unmatched
from step_3_candidate_matching import step_3_candidate_matching
from step_4_ai_candidate_selection import step_4_ai_candidate_selection
from step_5_ai_judgment import step_5_ai_judgment
from step_6_final_arbitration import step_6_ai_arbitration
from generate_final_report import generate_comprehensive_report

# ==============================================================================
# UI (用户界面) 定义
# ==============================================================================
app_ui = ui.page_navbar(
    shinyswatch.theme.pulse(), # 使用一个现代主题
    ui.nav("🏠 主页 & 配置",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("⚙️ 核心参数配置"),
                ui.input_slider("max_workers", "最大并发数", min=1, max=50, value=Config.MAX_WORKERS),
                ui.input_slider("candidate_limit", "模糊匹配候选数量", min=5, max=50, value=Config.CANDIDATE_LIMIT),
                ui.hr(),
                ui.h4("📁 文件路径配置"),
                ui.input_text("raw_parquet_file", "原始数据文件路径", value=Config.RAW_PARQUET_FILE),
                ui.input_text("site_dict_file", "机构字典文件路径", value=Config.SITE_DICT_FILE),
                ui.input_text("results_dir", "结果输出目录", value=Config.RESULTS_DIR),
            ),
            ui.h2("🏥 医疗机构名称智能匹配与标准化系统"),
            ui.markdown("欢迎使用！请在左侧侧边栏配置您的API信息和文件路径。"),
            ui.hr(),
            ui.h4("🔑 API 与模型配置"),
            ui.input_text("openai_base_url", "OpenAI API Base URL", value=Config.OPENAI_BASE_URL),
            ui.input_password("openai_api_key", "OpenAI API Key", value=""),
            ui.input_text("translate_model", "翻译模型", value=Config.TRANSLATE_MODEL),
            ui.hr(),
            ui.input_text("genai_base_url", "Google GenAI Base URL", value=Config.GENAI_BASE_URL),
            ui.input_password("genai_api_key", "Google GenAI API Key", value=""),
            ui.row(
                ui.column(4, ui.input_text("select_model", "选择模型", value=Config.AI_SELECT_MODEL)),
                ui.column(4, ui.input_text("judge_model", "判断模型", value=Config.AI_JUDGE_MODEL)),
                ui.column(4, ui.input_text("arbitrate_model", "仲裁模型", value=Config.ARBITRATE_MODEL)),
            )
        )
    ),
    ui.nav("📝 Prompt 编辑器",
        ui.h2("📝 Prompt 编辑器"),
        ui.markdown("在这里，您可以实时查看和修改用于AI处理的系统提示词 (Prompt)。"),
        ui.accordion(
            ui.accordion_panel("Step 2: 翻译 Prompt", ui.input_text_area("prompt_translate", "", value=BATCH_TRANSLATE_PROMPT, height="400px")),
            ui.accordion_panel("Step 4: AI选择 Prompt", ui.input_text_area("prompt_select", "", value=BATCH_AI_SELECT_PROMPT, height="400px")),
            ui.accordion_panel("Step 5: AI判断 Prompt", ui.input_text_area("prompt_judge", "", value=BATCH_JUDGE_PROMPT, height="400px")),
            ui.accordion_panel("Step 6: AI仲裁 Prompt", ui.input_text_area("prompt_arbitrate", "", value=BATCH_ARBITRATE_PROMPT, height="400px")),
        )
    ),
    ui.nav("🚀 执行 & 结果",
        ui.h2("🚀 执行流水线 & 查看结果"),
        ui.input_checkbox_group(
            "steps_to_run",
            "选择要执行的步骤:",
            {
                "step1": "Step 1: 英文精确匹配", "step2": "Step 2: AI 翻译",
                "step3": "Step 3: 候选匹配", "step4": "Step 4: AI 候选选择",
                "step5": "Step 5: AI 判断", "step6": "Step 6: AI 最终仲裁",
                "step7": "Step 7: 生成最终报告"
            },
            selected=["step1", "step2", "step3", "step4", "step5", "step6", "step7"],
            inline=True
        ),
        ui.input_action_button("run_pipeline", "🚀 执行所选步骤", class_="btn-primary"),
        ui.hr(),
        ui.h4("📜 执行日志"),
        ui.output_text_verbatim("run_log"),
        ui.hr(),
        ui.h4("📊 最终报告"),
        ui.output_ui("report_display")
    ),
    title="医疗机构匹配系统"
)

# ==============================================================================
# Server (后台逻辑) 定义
# ==============================================================================
def server(input, output, session):
    
    # --- 响应式变量 ---
    # 创建一个响应式的值来存储日志
    log_messages = reactive.Value(["### 🚀 流水线执行日志\n\n"])
    
    @reactive.Calc
    def get_dynamic_config():
        """创建一个动态的配置对象，它会响应UI上的任何变化"""
        config = Config()
        # 从UI输入更新配置
        config.MAX_WORKERS = input.max_workers()
        config.CANDIDATE_LIMIT = input.candidate_limit()
        config.RAW_PARQUET_FILE = input.raw_parquet_file()
        config.SITE_DICT_FILE = input.site_dict_file()
        config.RESULTS_DIR = input.results_dir()
        config.OPENAI_BASE_URL = input.openai_base_url()
        config.OPENAI_API_KEY = input.openai_api_key()
        config.TRANSLATE_MODEL = input.translate_model()
        config.GENAI_BASE_URL = input.genai_base_url()
        config.GENAI_API_KEY = input.genai_api_key()
        config.AI_SELECT_MODEL = input.select_model()
        config.AI_JUDGE_MODEL = input.judge_model()
        config.ARBITRATE_MODEL = input.arbitrate_model()
        
        # 更新 Prompts
        config.BATCH_TRANSLATE_PROMPT = input.prompt_translate()
        config.BATCH_AI_SELECT_PROMPT = input.prompt_select()
        config.BATCH_JUDGE_PROMPT = input.prompt_judge()
        config.BATCH_ARBITRATE_PROMPT = input.prompt_arbitrate()
        
        return config

    @output
    @render.text
    def run_log():
        """渲染日志输出区域"""
        return "".join(log_messages())

    @reactive.Effect
    @reactive.event(input.run_pipeline)
    def _():
        """当“执行”按钮被点击时，触发此函数"""
        dynamic_config = get_dynamic_config()
        os.makedirs(dynamic_config.RESULTS_DIR, exist_ok=True)
        
        # 重置日志
        log_messages.set(["### 🚀 流水线执行日志\n\n"])

        def run_step(step_func, step_name):
            new_log = log_messages()
            new_log.append(f"**{datetime.now().strftime('%H:%M:%S')} - 正在执行 {step_name}...**\n")
            log_messages.set(new_log)
            
            success = step_func(dynamic_config) # 假设函数返回 True/False
            
            new_log = log_messages()
            if success:
                new_log.append(f"**{datetime.now().strftime('%H:%M:%S')} - ✅ {step_name} 完成！**\n\n")
            else:
                new_log.append(f"**{datetime.now().strftime('%H:%M:%S')} - ❌ {step_name} 失败！流水线终止。**\n\n")
            log_messages.set(new_log)
            return success

        pipeline_steps = {
            "step1": ("Step 1: 英文精确匹配", step_1_initial_english_match),
            "step2": ("Step 2: AI 翻译", step_2_translate_unmatched),
            "step3": ("Step 3: 候选匹配", step_3_candidate_matching),
            "step4": ("Step 4: AI 候选选择", step_4_ai_candidate_selection),
            "step5": ("Step 5: AI 判断", step_5_ai_judgment),
            "step6": ("Step 6: AI 最终仲裁", step_6_ai_arbitration),
            "step7": ("Step 7: 生成最终报告", generate_comprehensive_report)
        }
        
        # 在一个新线程中运行，防止UI被阻塞
        def pipeline_thread():
            for step_key in input.steps_to_run():
                step_name, step_func = pipeline_steps[step_key]
                if not run_step(step_func, step_name):
                    break
        
        thread = Thread(target=pipeline_thread)
        thread.start()

    @output
    @render.ui
    def report_display():
        """渲染最终的HTML报告"""
        # 依赖于“执行”按钮，并且在报告文件存在时自动刷新
        input.run_pipeline() 
        
        report_path = get_dynamic_config().FINAL_REPORT_OUTPUT_FILE
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return ui.HTML(f'<iframe srcdoc="{html_content.replace("\"", "&quot;")}" width="100%" height="800px" style="border:none;"></iframe>')
        else:
            return ui.p("尚未生成报告。请选择步骤并点击“执行”。")

# ==============================================================================
# App 实例化
# ==============================================================================
app = App(app_ui, server)