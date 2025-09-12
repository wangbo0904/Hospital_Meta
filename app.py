# /SITE/app.py

import shutil
from pathlib import Path
from shiny import App, render, ui, reactive, req
import shinyswatch
import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

# 导入您的模块
import config_utils
from config_utils import Config, logger
import step_1_english_match
import step_2_translate
import step_3_candidate_matching
import step_4_ai_candidate_selection
import step_5_ai_judgment
import step_6_final_arbitration
import step_7_generate_final_report

# --- UI 定义 --- 
app_ui = ui.page_fluid(
    # ui.include_css(Path(__file__).parent / "www/style.css"),

    # 主体内容
    ui.div(
        {"class": "container-fluid"},
        ui.h2("PI-SITE 匹配流水线"),
        ui.layout_sidebar(
            # 侧边栏内容
            ui.sidebar(
                ui.div(
                    {"class": "container"},
                    ui.h4("参数配置"),
                    ui.input_file("raw_data_file", "上传原始数据 (例如 test.parquet)", accept=".parquet"),
                    ui.input_file("dict_file", "上传字典文件 (例如 site_dict.parquet)", accept=".parquet"),
                    ui.hr(),
                    ui.h4("AI 模型设置"),
                    ui.input_text("organization", "Organization", value=config_utils.Config().ORGANIZATION),
                    ui.input_select("translate_model", "翻译模型", {"deepseek": "deepseek", "global-gemini-2.5-pro": "global-gemini-2.5-pro"}),
                    ui.input_action_button("start_processing", "开始处理流水线", class_="btn-primary w-100")
                ),
                width=3
            ),
            
            # 主内容区域
            ui.div(
                {"class": "card"},
                ui.h4("实时进度条"),
                ui.output_ui("progress_bar"),
            ),
            ui.div(
                {"class": "card"},
                ui.h4("处理日志"),
                ui.div(ui.output_text_verbatim("log_output", placeholder=True), id="run_log"),
            ),
            ui.div(
                {"class": "card"},
                ui.h4("最终报告"),
                ui.div(ui.output_ui("final_report_display"), id="report_display"),
            ),
        ),
    ),
    
    title="PI-SITE Matching Pipeline",
    theme=shinyswatch.theme.darkly,  # 启用 darkly 主题
)

# --- Server Logic --- 
def server(input, output, session):
    log_messages = reactive.Value("")
    report_path = reactive.Value("")
    current_step = reactive.Value(0)
    
    @reactive.Effect
    @reactive.event(input.start_processing)
    def _():
        raw_file = input.raw_data_file()
        dict_file = input.dict_file()
        
        if not raw_file or not dict_file:
            msg = "错误：请上传原始数据和字典文件。"
            log_messages.set(msg)
            logger.error(msg)
            return
        
        log_messages.set("开始处理...\n")
        report_path.set("")
        current_step.set(0)
        config = Config()

        for file_info in [raw_file[0], dict_file[0]]:
            if not file_info['name'].endswith('.parquet'):
                msg = f"错误：文件 {file_info['name']} 不是有效的'.parquet'格式。"
                log_messages.set(msg)
                logger.error(msg)
                return
            
            upload_path = config.DATA_DIR / file_info['name']
            shutil.move(file_info['datapath'], upload_path)
            log_messages.set(log_messages() + f"已保存上传文件到：{upload_path}\n")

        ui_inputs = {
            "organization": input.organization(),
            "translate_model": input.translate_model(),
            "raw_data_file": raw_file[0],
            "dict_file": dict_file[0]
        }
        config.update_from_ui(ui_inputs)
        
        log_messages.set(log_messages() + "配置已更新，开始流水线...\n")
        run_pipeline(config)

    def run_pipeline(config_instance):
        steps = [
            ("Step 1: Initial English Match", step_1_english_match.step_1_initial_english_match),
            ("Step 2: Translate Unmatched", step_2_translate.step_2_translate_unmatched),
            ("Step 3: Candidate Matching", step_3_candidate_matching.step_3_candidate_matching),
            ("Step 4: AI Candidate Selection", step_4_ai_candidate_selection.step_4_ai_candidate_selection),
            ("Step 5: AI Judgment", step_5_ai_judgment.step_5_ai_judgment),
            ("Step 6: Final AI Arbitration", step_6_final_arbitration.step_6_ai_arbitration),
            ("Step 7: Generate Final Report", step_7_generate_final_report.generate_comprehensive_report),
        ]
        
        total_steps = len(steps)
        try:
            for i, (name, func) in enumerate(steps):
                current_step.set(i + 1)
                log_messages.set(log_messages() + f"--- 正在运行: {name} ---\n")
                success = func(config_instance)
                if not success:
                    error_msg = f"错误：{name} 失败，流水线中止。"
                    log_messages.set(log_messages() + error_msg + "\n")
                    logger.error(error_msg)
                    return
                log_messages.set(log_messages() + f"--- 完成: {name} ---\n")
            
            log_messages.set(log_messages() + "流水线成功完成！\n")
            report_path.set(str(config_instance.FINAL_REPORT_OUTPUT_FILE))
            current_step.set(total_steps)
        except Exception as e:
            error_msg = f"发生严重错误：{e}"
            log_messages.set(log_messages() + error_msg + "\n")
            logger.critical(error_msg, exc_info=True)

    @output
    @render.text
    def log_output():
        return log_messages()[-1000:]

    @output
    @render.ui
    def progress_bar():
        step = current_step()
        if step == 0:
            return ui.p("等待开始...")
        progress = (step / 7) * 100
        return ui.HTML(f"""
            <div class="progress" style="height: 20px; margin-bottom: 1rem;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: {progress}%; background-color: #007bff;">
                    {step}/7 步骤完成 ({progress:.1f}%)
                </div>
            </div>
        """)

    @output
    @render.ui
    def final_report_display():
        path = report_path()
        req(path)
        
        if not os.path.exists(path):
            return ui.p(f"报告文件不存在：{path}", style="color:red;")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return ui.HTML(f'<iframe srcdoc="{html_content.replace("\"", "&quot;")}" style="width: 100%; height: 80vh; border: none;"></iframe>')
        except Exception as e:
            return ui.p(f"读取报告文件时出错：{e}", style="color:red;")

app = App(app_ui, server)
