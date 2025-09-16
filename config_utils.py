# /SITE/config_utils.py

import os
import pandas as pd
import json
from pathlib import Path
import threading
from queue import Queue, Empty
from typing import List, Dict, Tuple
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局关闭事件
shutdown_event = threading.Event()

# --- 提示词 (全局加载，以便其他模块可以导入) ---
BATCH_CLASSIFY_PROMPT = """
**# Role**
You are an expert data validator specializing in medical and academic affiliations from mainland China. Your task is to classify raw affiliation strings into predefined categories based on their content and structure.

**# Core Task**
For each input string, you must determine if it represents a specific, identifiable medical institution or if it falls into a category of abnormal/non-institutional data.

**# Classification Categories**
You must assign one of the following exact category tags to each input:

1.  **`Valid Institution`**: The string clearly refers to a specific hospital, medical center, or a university that directly implies a medical school/hospital (e.g., "Peking University"). It should be a concrete, verifiable entity.
2.  **`Personal Name`**: The string is clearly a person's name (Chinese or English).
    *   *Examples*: "Min Huang", "Zhang San"
3.  **`Generic Term`**: The string is a general, non-specific type of institution, lacking a unique geographical or institutional identifier.
    *   *Examples*: "Maternal and Children Health Care Hospital", "Cancer Hospital", "People's Hospital"
4.  **`University (Non-Medical Focus)`**: The string refers to a university but does not specify a hospital, medical school, or a department, and is not a university known primarily for medicine.
    *   *Examples*: "Chongqing Medical University", "Tsinghua University" (Note: While it has a hospital, the string itself is just the university).
5.  **`Department/Ward`**: The string refers **only** to a clinical department or ward, **without mentioning a specific parent institution**.
    *   *Examples*: "肿瘤科" (Oncology Department), "Internal Medicine", "Surgical Ward", **"Department of Vascular Ultrasonography"**
6.  **`Contact Info`**: The string is an email address, phone number, or website URL.
    *   *Examples*: "huanghq@sysucc.org.cn", "86-10-12345678"
7.  **`Address`**: The string is a geographical address without a clear institution name.
    *   *Examples*: "No. 17, Fucheng Road, Haidian District, Beijing"
8.  **`Irrelevant/Other`**: The string is nonsensical, junk data, or clearly not an affiliation.
    *   *Examples*: "Not applicable", "---", "12345", "None"

**# Output Format**
Strictly return a single JSON object with one key: `"classifications"`.
The value must be an array of objects, each containing:
*   `"esid"`: The unique numeric ID from the input record. You must copy this value verbatim.
*   `"affiliation"`: The original, unmodified input string.
*   `"category"`: One of the exact category tags listed above.
*   `"reason"`: A brief Chinese explanation for your classification.

**# Example**
**User Input:**
```json
{
  "inputs": [
    {
      "esid": 1001,
      "affiliation": "The First Affiliated Hospital of Sun Yat-sen University"
    },
    {
      "esid": 1002,
      "affiliation": "Min Huang"
    },
    {
      "esid": 1003,
      "affiliation": "Maternal and Children Health Care Hospital"
    },
    {
      "esid": 1004,
      "affiliation": "Chongqing Medical University"
    },
    {
      "esid": 1005,
      "affiliation": "huanghq@sysucc.org.cn"
    },
    {
      "esid": 1006,
      "affiliation": "肿瘤科"
    }
  ]
}
```

Your Expected Output:
```json
{
  "classifications": [
    {
      "esid": 1001,
      "affiliation": "The First Affiliated Hospital of Sun Yat-sen University",
      "category": "Valid Institution",
      "reason": "这是一个完整的、可识别的医疗机构名称。"
    },
    {
      "esid": 1002,
      "affiliation": "Min Huang",
      "category": "Personal Name",
      "reason": "这是一个典型的人名格式。"
    },
    {
      "esid": 1003,
      "affiliation": "Maternal and Children Health Care Hospital",
      "category": "Generic Term",
      "reason": "这是一个通用机构类型，缺少具体的地理或机构限定词。"
    },
    {
      "esid": 1004,
      "affiliation": "Chongqing Medical University",
      "category": "University (Non-Medical Focus)",
      "reason": "这是一个大学名称，而非其附属医院。"
    },
    {
      "esid": 1005,
      "affiliation": "huanghq@sysucc.org.cn",
      "category": "Contact Info",
      "reason": "这是一个标准的电子邮件地址格式。"
    },
    {
      "esid": 1006,
      "affiliation": "肿瘤科",
      "category": "Department/Ward",
      "reason": "这是一个临床科室名称。"
    }
  ]
}
```
"""

BATCH_TRANSLATE_PROMPT = """
**# Role**
You are a top-tier medical information analysis expert and detective. Your core mission is to use rigorous web investigation to find the **most official and standardized full Chinese name** for each institution.

**# Core Principles (Your Guiding Philosophy)**
1.  **Evidence is Paramount**: Your final translation must be based on verifiable evidence (e.g., official websites, government directories, authoritative articles).
2.  **Geographical Context as a Clue, Not a Rule**: The provided `state` and `city` fields are **strong clues** to help you disambiguate, but they might be inaccurate. **Do not infer a hospital's name solely based on the provided location.** The primary evidence must come from matching the `original_input` name itself.
3.  **Strict Translation Rules**: You must follow these specific translation patterns:
    *   `"People's Hospital"` must be translated to **"人民医院"**.
    *   `"Center Hospital"` or `"Central Hospital"` must be translated to **"中心医院"**.
    *   `"Municipal Hospital"` must be translated to **"市立医院"**.

**# Task and Workflow: An Expert's Verification Process**
For each input, find the best possible Chinese name.

**Step 1: Initial Analysis & Rule-Based Translation.**
   - Deconstruct the `original_input` to identify the core entity.
   - Apply the **Strict Translation Rules** first. For example, if you see "People's Hospital", immediately translate that part to "人民医院".

**Step 2: Evidence Gathering with Geographical Clues.**
   - Formulate web search queries using the core entity and the provided `state` and `city` as helpful context.
   - **Example**: For `original_input: "First People's Hospital", city: "Yulin"`, your search should be `"第一人民医院" 玉林` and also `"First People's Hospital" Yulin`.

**Step 3: Evidence Evaluation and Final Confirmation.**
   - Evaluate the search results from authoritative sources.
   - **If you find Tier 1 or Tier 2 evidence** that directly links the original_input to a specific Chinese name, use that name with High confidence.
   - **If evidence is weak or non-existent**, construct the most plausible name based on structure and pinyin, but use Low confidence.

**# Example**

**User Input:**
```json
{
  "inputs": [
    {
      "esid": f8c95c69c563c86ec576b0ce4e660cfd,
      "original_input": "The First Hospital Of Yulin",
      "state": "Shaanxi",
      "city": "Yulin"
    },
    {
      "esid": f8c9471d2f64263e1c9b4761ce4a1203,
      "original_input": "Yulin City first people's hospital",
      "state": "Guangxi Zhuang Autonomous Region",
      "city": "Yulin"
    },
    {
      "esid": f8c944ab49926d47c96c86c3b0164166,
      "original_input": "Anfu Center Hospital",
      "state": "Jiangxi",
      "city": "Ji'an"
    }
  ]
}

Your Expected Output:
```json
{
  "translations": [
    {
      "esid": f8c95c69c563c86ec576b0ce4e660cfd,
      "original_input": "The First Hospital Of Yulin",
      "state": "Shaanxi",
      "city": "Yulin",
      "original_input_zh": "榆林市第一医院",
      "trans_confidence": "High"
    },
    {
      "esid": f8c9471d2f64263e1c9b4761ce4a1203,
      "original_input": "Yulin City first people's hospital",
      "state": "Guangxi Zhuang Autonomous Region",
      "city": "Yulin",
      "original_input_zh": "玉林市第一人民医院",
      "trans_confidence": "High"
    },
    {
      "esid": f8c944ab49926d47c96c86c3b0164166,
      "original_input": "Anfu Center Hospital",
      "state": "Jiangxi",
      "city": "Ji'an",
      "original_input_zh": "安福县中心医院",
      "trans_confidence": "Low"
    }
  ]
}
```
"""

BATCH_AI_SELECT_PROMPT = """
**# Role**
You are a **Senior Medical Institution Specialist** for mainland China. Your expertise lies in identifying and verifying hospital names, understanding their official designations, common aliases, historical changes, and organizational structures. You are not a simple matcher; you are an expert providing an authoritative judgment based on verified evidence.

**# Core Principles (Your Guiding Philosophy)**
1.  **Evidence is Paramount**: Your judgment must be based on verifiable evidence. For any non-exact match, your default action is to perform a web search. A match is only valid if supported by authoritative sources.
2.  **Authoritative Source Hierarchy**: You must prioritize evidence from the most reliable medical and governmental sources:
    *   **Tier 1 (Highest)**: Official hospital websites (especially the "About Us" or "Contact" pages) and official hospital directories from government Health Commissions (卫健委).
    *   **Tier 2**: Major, reliable map providers (e.g., Baidu Maps, Gaode Maps) that show official location data.
    *   **Tier 3**: Publications in reputable academic journals or reports from major, official news outlets.
3.  **Understand Naming Conventions**: You are an expert in how Chinese hospitals are named. You know that:
    *   **Geographical Context is Crucial**: Searches must include `state` and `city` to differentiate hospitals like "榆林市第一医院" (Shaanxi) from "玉林市第一人民医院" (Guangxi).
    *   **Common Variations Exist**: Terms like "人民医院" vs. "第一人民医院" or "中医院" vs. "中医医院" often refer to the same entity, but this **must be verified** for each specific city, as exceptions exist.
    *   **English Names are Key Evidence**: A strong match between the input `affiliation` and a candidate's `hospital_name_en` is a critical piece of supporting evidence.

**# Task and Workflow: An Expert's Verification Process**
For each input, you must conduct an expert review to find the correct match from the `candidates` list.

**Step 1: Initial Assessment & Hypothesis.**
   - Review the input (`affiliation`, `affiliation_cn`, `state`, `city`).
   - Scan the candidate list for plausible matches based on name similarity and location.
   - Formulate a hypothesis. Example: "Hypothesis: The input '聊城市第一人民医院' is the same entity as the candidate '聊城市人民医院'."

**Step 2: Evidence Gathering & Synthesis.**
   - **Action**: **Execute web searches to gather evidence from various sources.** Your goal is to either confirm or disprove your hypothesis.
   - **Synthesize Evidence**: Analyze the search results according to the **Authoritative Source Hierarchy**. Does Tier 1 or Tier 2 evidence confirm the hypothesis? Is there conflicting information?

**Step 3: Evidence Synthesis & Strict Confidence Assessment.**
   - Analyze the search results and assign a confidence level based on the **following strict criteria**:
      *   **High**: You found **direct, authoritative evidence (Tier 1 or Tier 2)** that explicitly confirms the input name and the candidate name are equivalent. This includes:
          - An exact name match.
          - An official alias listed on the hospital's website or a government directory.
          - A common variation (e.g., "第一人民医院" vs. "人民医院") that is **explicitly confirmed** by an authoritative source for that specific city.
      *   **Low**: The names are **highly similar**, and the locations match, but you **could not find any direct, authoritative evidence** to confirm they are the same entity. The match is a plausible inference based on name structure alone.
      *   **No Match**: Evidence proves the names are different, or the name similarity is not high enough to be plausible.

**Step 4: Final Selection and Reporting.**
   - Select the candidate with the highest confidence level. If multiple candidates have the same confidence, choose the one with the highest name similarity.
   - If no candidate reaches at least "Low" confidence, return `null`.
   - Your `reason` must justify your selection and the assigned confidence level.

**# Output Format and Structure (JSON) - Must be strictly followed**
Your output must be a single JSON object containing a `results` list.

Each result object **must** contain the following ten keys:

*   `"esid"`: [Integer] - The unique numeric ID for the input record.
*   `"affiliation"`: [String] - The original English or Pinyin institutional name from the input.
*   `"affiliation_cn"`: [String] - The original Chinese institutional name from the input.
*   `"state"`: [String] - The original province/state information from the input.
*   `"city"`: [String] - The original city information from the input.

*   **--- Matched Candidate Information ---**
    *   **If a confident match is found**: You must populate the following four fields with the information from the **single selected candidate object**.
        *   `"matched_site"`: [String] - The `hospital_name` of the selected candidate.
        *   `"matched_site_en"`: [String] - The `hospital_name_en` of the selected candidate.
        *   `"matched_province"`: [String] - The `hospital_province` of the selected candidate.
        *   `"matched_city"`: [String] - The `hospital_city` of the selected candidate.
    *   **【KEY INSTRUCTION】If confidence is No Match**: **All four** of these fields (`matched_site`, `matched_site_en`, `matched_province`, `matched_city`) **must be `null`**.
*   `"confidence"`: [String] - Your assessed confidence level. Must be one of: **"High"**, **"Low"**, or **"No Match"**.
*   `"reason_tag"`: [String] - A brief, standardized Chinese label summarizing the reason.
*   `"reason"`: [String] - In Chinese, explaining the connection to the selected hospital_name, referencing `hospital_name_en` if relevant.

**# Example Output:**
```json
{
  "results": [
    {
      "esid": 12345,
      "affiliation": "The First People's Hospital of Liaocheng",
      "affiliation_cn": "聊城市第一人民医院",
      "state": "Shandong",
      "city": "Liaocheng",
      "matched_site": "聊城市人民医院",
      "matched_site_en": "Liaocheng People's Hospital",
      "matched_province": "Shandong",
      "matched_city": "Liaocheng",
      "confidence": "High",
      "reason_tag": "常见变体(已验证)",
      "reason": "候选'聊城市人民医院'地理位置一致。经网络搜索，其官网明确提及'聊城市第一人民医院'是其官方名称之一。存在直接的Tier 1证据，因此置信度评为'高'。"
    },
    {
      "esid": 67890,
      "affiliation": "...",
      "affiliation_cn": "江苏大学附属第三医院",
      "state": "Jiangsu",
      "city": "Zhenjiang",
      "matched_site": "江苏大学附属医院",
      "matched_site_en": "...",
      "matched_province": "Jiangsu",
      "matched_city": "Zhenjiang",
      "confidence": "Low",
      "reason_tag": "高度相似(待确认)",
      "reason": "候选'江苏大学附属医院'地理位置一致。名称高度相似，但网络搜索未能找到任何权威证据来直接证实'江苏大学附属医院'就是第三附属医院。这是一个基于名称相似性的合理推断，但缺乏证据支持，因此置信度评为'低'。"
    }
  ]
}
```
"""
BATCH_JUDGE_PROMPT = """
**# Role**
You are a professional medical data governance assistant, specializing in determining if two medical institutions refer to the same entity through name analysis and web searches.

**# Important Directives:**
1.  **Strictly Prohibit Modifying Input Names**: Base your judgment on the originally provided names. Do not alter or "correct" any part of the names.
2.  **Output Names Verbatim**: Maintain the input `pi_site_name` and `matched_site` values exactly as provided.
3.  **Base Judgment on Original Names**: Do not assume a name is incorrect or modify it.

**# Task:**
Compare a pair of medical institution names to determine if they represent the same entity. Use reasoning based on medical industry knowledge and web searches for uncertain cases.

**# Analysis Steps:**
1.  **Initial Name Analysis**: Judge based on name similarity.
2.  **Web Search Verification**: For similar but uncertain names, verify using your knowledge base.
3.  **Comprehensive Judgment**: Combine name analysis and web search results.

**# Judgment Criteria (Same entity if ANY met):**
1. **Obvious Typographical Error**: Clear typos (e.g., "上栗县中医医医院" vs. "上栗县中医医院" -> `is_same: true`).
2. **Common Name Variation**: Interchangeable names (e.g., "中医医院" vs. "中医院").
3. **Exact Name Match**: No verification needed.
4. **Abbreviation and Full Name**: E.g., "北医三院" vs. "北京大学第三医院".
5. **Inclusion Relationship**: E.g., "邵逸夫医院" vs. "浙江大学医学院附属邵逸夫医院".
6. **Recognized Alias/Former Name**: E.g., "北京协和医院" vs. "北京协和医学院附属医院".
7. **Obvious Acronyms/Shortenings**: E.g., "华西医院" vs. "四川大学华西医院".
8. **Web Search Confirmation**: Both names point to the same hospital.

**# Output Format:**
```json
{
  "results": [
    {
      "esid": 54321,
      "pi_site_name": "北医三院",
      "matched_site": "北京大学第三医院",
      "is_same": true,
      "reason": "名称包含关系，经搜索验证确认'北医三院'是'北京大学第三医院'的官方简称"
    }
  ]
}
```
"""

class Config:
    def __init__(self):
        # --- 基础路径 (相对于脚本位置) ---
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.OUTPUT_REPORT_DIR = self.BASE_DIR / "reports"

        # 确保目录存在
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.OUTPUT_REPORT_DIR.mkdir(exist_ok=True)
        
        # --- 文件路径 (占位符，将由应用动态更新) ---
        self.RAW_PARQUET_FILE = self.DATA_DIR / "test.parquet"
        self.SITE_DICT_FILE = self.DATA_DIR / "site_dict_0911.parquet"
        
        # --- 中间过程与输出文件 ---
        self.CLASSIFY_JSONL_FILE = self.RESULTS_DIR / "step0_classification_output.jsonl"
        self.NORMAL_DATA_FILE = self.RESULTS_DIR / "step0_normal_data.parquet"
        self.ABNORMAL_DATA_FILE = self.RESULTS_DIR / "step0_abnormal_data.parquet"
        self.MATCHED_EN_FILE = self.RESULTS_DIR / "step1_matched_en.parquet"
        self.UNMATCHED_EN_FILE = self.RESULTS_DIR / "step1_unmatched_en.parquet"
        self.TRANSLATED_JSONL_FILE = self.RESULTS_DIR / "step2_translated.jsonl"
        self.META_MATCH_JSON_FILE = self.RESULTS_DIR / "step3_meta_match_results.json"
        self.META_PARTIAL_EXACT_MATCHED_FILE = self.RESULTS_DIR / "step3_meta_partial_exact_matched.parquet"
        self.AI_META_INPUT_FILE = self.RESULTS_DIR / "step3_ai_meta_input.parquet"
        self.AI_META_OUTPUT_JSONL_FILE = self.RESULTS_DIR / "step4_ai_meta_output.jsonl"
        self.ALL_MATCHES_COMBINED_FILE = self.RESULTS_DIR / "step4_all_matches_combined.parquet"
        self.JUDGE_INPUT_FILE = self.RESULTS_DIR / "step4_judge_input.parquet"
        self.FINAL_JUDGE_OUTPUT_JSONL_FILE = self.RESULTS_DIR / "step5_final_judge_output.json"
        self.FINAL_JUDGE_OUTPUT_PARQUET_FILE = self.RESULTS_DIR / "step5_final_judge_output.parquet"
        self.FINAL_REPORT_TEMPLATE_FILE = self.BASE_DIR / "template/final_report_template.html"
        self.FINAL_REPORT_TEMPLATE_NO_STEP5_FILE = self.BASE_DIR / "template/final_report_template_no_step5.html"
        self.FINAL_REPORT_OUTPUT_FILE = self.OUTPUT_REPORT_DIR / "final_comprehensive_report.html"
        self.FINAL_OUTPUT = self.RESULTS_DIR / "step6_final_output.parquet"

        # --- 来自您文件的API和模型设置 ---
        self.OPENAI_BASE_URL = "http://116.63.133.80:30660/api/llm/v1"
        self.OPENAI_API_KEY = "5YWs5YWx5pWw5o2uLeS4tOW6ig=="
        self.GENAI_BASE_URL = "https://globalai.vip/"
        self.GENAI_API_KEY = "sk-pF9rUA3j4igwJbP0xQN2izR6jwQGY0ke4xXKBQnUdkHCZtF9"
        self.API_PROJECT = "PI_SITE"
        self.ORGANIZATION = "WB"
        self.CLASSIFY_MODEL = "global-gemini-2.5-flash"
        self.TRANSLATE_MODEL = "gemini-2.5-flash-nothinking"
        self.AI_SELECT_MODEL = "gemini-2.5-flash-nothinking"
        self.AI_JUDGE_MODEL = "gemini-2.5-flash-nothinking"

        # --- 来自您文件的性能和速率限制设置 ---
        self.MAX_WORKERS = 30
        self.CALLS_PER_MINUTE = 300
        self.ONE_MINUTE = 60
        self.RETRY_ATTEMPTS = 3

        self.CLASSIFY_BATCH_SIZE = 100
        self.TRANSLATE_BATCH_SIZE = 50
        self.AI_SELECT_BATCH_SIZE = 10
        self.AI_JUDGE_BATCH_SIZE = 10
        self.ARBITRATE_BATCH_SIZE = 10
        self.CANDIDATE_LIMIT = 20

    def update_from_ui(self, ui_inputs: dict):
        """从 Shiny UI 输入更新配置属性"""
        logger.info("Updating configuration from UI...")
        self.ORGANIZATION = ui_inputs.get("organization", self.ORGANIZATION)
        self.TRANSLATE_MODEL = ui_inputs.get("translate_model", self.TRANSLATE_MODEL)
        self.AI_SELECT_MODEL = ui_inputs.get("ai_select_model", self.AI_SELECT_MODEL)
        self.AI_JUDGE_MODEL = ui_inputs.get("ai_judge_model", self.AI_JUDGE_MODEL)
        
        # 根据上传的文件名更新文件路径
        if "raw_data_file" in ui_inputs:
            self.RAW_PARQUET_FILE = self.DATA_DIR / ui_inputs["raw_data_file"]["name"]
        if "dict_file" in ui_inputs:
            self.SITE_DICT_FILE = self.DATA_DIR / ui_inputs["dict_file"]["name"]
        
        logger.info(f"Raw data file set to: {self.RAW_PARQUET_FILE}")
        logger.info(f"Dictionary file set to: {self.SITE_DICT_FILE}")

    def clear_results_dir(self):
        """删除 results 目录中的所有文件，以确保一次干净的运行"""
        logger.info(f"Clearing previous results from {self.RESULTS_DIR}...")
        for file_path in self.RESULTS_DIR.glob('*'):
            try:
                if file_path.is_file():
                    file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

# --- 工具函数 (来自您的文件) ---
def is_chinese(text: str) -> bool:
    """检查字符串是否包含中文字符"""
    if not isinstance(text, str): return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def async_saver_local(output_filepath: str, local_queue: Queue):
    with open(output_filepath, 'a', encoding='utf-8') as f:
        while True:
            try:
                result = local_queue.get(timeout=1)
                if result is None:
                    local_queue.task_done()
                    break
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()
                local_queue.task_done()
            except Empty:
                if shutdown_event.is_set(): break
                continue
            except Exception as e:
                logger.error(f"Async save failed: {str(e)}")

def get_tasks_to_process_generic(input_df: pd.DataFrame, id_column: str, output_jsonl_path: str, output_id_column: str = None) -> Tuple[List[Dict], pd.DataFrame]:
    if output_id_column is None: output_id_column = id_column
    if id_column not in input_df.columns: raise ValueError(f"Error: ID column '{id_column}' not found in input DataFrame.")
    
    input_df = input_df.drop_duplicates(subset=[id_column]).reset_index(drop=True)
    source_tasks_map = {row[id_column]: row.to_dict() for _, row in input_df.iterrows()}
    
    processed_task_ids = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if data.get('status') == 'success' and output_id_column in data:
                        processed_task_ids.add(data[output_id_column])
                except json.JSONDecodeError:
                    continue
    
    tasks_to_run_ids = set(source_tasks_map.keys()) - processed_task_ids
    tasks_to_run = [source_tasks_map[task_id] for task_id in tasks_to_run_ids]
    logger.info(f"Validation complete. Total: {len(source_tasks_map)}, Processed: {len(processed_task_ids)}, To-Do: {len(tasks_to_run)}")
    return tasks_to_run, input_df

def cleanup_jsonl_file(filepath: str, id_column: str):
    if not os.path.exists(filepath): return
    all_records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try: all_records.append(json.loads(line))
            except json.JSONDecodeError: continue
    if not all_records: return
    df = pd.DataFrame(all_records)
    if id_column not in df.columns: return
    df_latest = df.drop_duplicates(subset=[id_column], keep='last')
    df_clean = df_latest[df_latest['status'] == 'success'].copy()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(df_clean.to_json(orient='records', lines=True, force_ascii=False))