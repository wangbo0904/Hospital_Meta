# config_utils.py
import os
import pandas as pd
import json
import re
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

class Config:
    # 文件路径
    RAW_PARQUET_FILE = "./SITE_APP/data/test.parquet"
    SITE_DICT_FILE = "./SITE_APP/data/site_dict_0911.parquet"
    RESULTS_DIR = "./SITE_APP/results/"
    OUTPUT_REPORT_DIR = "./SITE_APP/reports/"
    MATCHED_EN_FILE = os.path.join(RESULTS_DIR, "step1_matched_en.parquet")
    UNMATCHED_EN_FILE = os.path.join(RESULTS_DIR, "step1_unmatched_en.parquet")
    TRANSLATED_JSONL_FILE = os.path.join(RESULTS_DIR, "step2_translated.jsonl")
    META_MATCH_JSON_FILE = os.path.join(RESULTS_DIR, "step3_meta_match_results.json")
    META_PARTIAL_EXACT_MATCHED_FILE = os.path.join(RESULTS_DIR, "step3_meta_partial_exact_matched.parquet")
    AI_META_INPUT_FILE = os.path.join(RESULTS_DIR, "step3_ai_meta_input.parquet")
    AI_META_OUTPUT_JSONL_FILE = os.path.join(RESULTS_DIR, "step4_ai_meta_output.jsonl")
    ALL_MATCHES_COMBINED_FILE = os.path.join(RESULTS_DIR, "step4_all_matches_combined.parquet")
    JUDGE_INPUT_FILE = os.path.join(RESULTS_DIR, "step4_judge_input.parquet")
    FINAL_JUDGE_OUTPUT_JSONL_FILE = os.path.join(RESULTS_DIR, "step5_final_judge_output.json")
    FINAL_JUDGE_OUTPUT_PARQUET_FILE = os.path.join(RESULTS_DIR, "step5_final_judge_output.parquet")
    ARBITRATION_OUTPUT_JSONL_FILE = os.path.join(RESULTS_DIR, "step6_arbitration_output.jsonl")
    ARBITRATION_OUTPUT_PARQUET_FILE = os.path.join(RESULTS_DIR, "step6_arbitration_output.parquet")
    FINAL_REPORT_TEMPLATE_FILE = "./SITE_APP/final_report_template.html" # 假设模板在根目录
    FINAL_REPORT_OUTPUT_FILE = os.path.join(OUTPUT_REPORT_DIR, "final_comprehensive_report.html")

    # API and model settings
    OPENAI_BASE_URL = "" 
    OPENAI_API_KEY = ""
    GENAI_BASE_URL = ""
    GENAI_API_KEY = ""
    API_PROJECT = ""
    ORGANIZATION = ""
    TRANSLATE_MODEL = ""
    AI_SELECT_MODEL = ""
    AI_JUDGE_MODEL = ""
    ARBITRATE_MODEL = ""

    # Performance and rate limiting
    MAX_WORKERS = 30
    CALLS_PER_MINUTE = 300
    ONE_MINUTE = 60
    RETRY_ATTEMPTS = 3
    TRANSLATE_BATCH_SIZE = 100
    AI_SELECT_BATCH_SIZE = 10
    AI_JUDGE_BATCH_SIZE = 10
    ARBITRATE_BATCH_SIZE = 10
    CANDIDATE_LIMIT = 20


# Prompts
BATCH_TRANSLATE_PROMPT = """
**# Role**
You are a top-tier medical information analysis expert and detective. Your core mission is to use rigorous web investigation to find the **most official and standardized full Chinese name** for each institution.

**# Core Principles (Your Guiding Philosophy)**
1.  **Evidence is Paramount**: Your final translation must be based on verifiable evidence (e.g., official websites, government directories, authoritative articles).
2.  **Geographical Context as a Clue, Not a Rule**: **【KEY UPDATE】** The provided `state` and `city` fields are **strong clues** to help you disambiguate, but they might be inaccurate. **Do not infer a hospital's name solely based on the provided location.** The primary evidence must come from matching the `original_input` name itself.
3.  **Strict Translation Rules**: You must follow these specific translation patterns:
    *   `"People's Hospital"` must be translated to **"人民医院"**.
    *   `"Center Hospital"` or `"Central Hospital"` must be translated to **"中心医院"**.
4.  **Label All Inferences**: If you cannot find direct, authoritative evidence for a translation and must rely on inference (e.g., from pinyin or structural translation), you **must** append `(推测)` to the end of the Chinese name.

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
   - **If you find Tier 1 or Tier 2 evidence** that directly links the `original_input` to a specific Chinese name, use that name.
   - **If evidence is weak or non-existent**, construct the most plausible name based on structure and pinyin, but remember to **append `(推测)`**.

**# Output Format**
(此部分保持不变)

**# Example**

**User Input:**
```json
{
  "inputs": [
    {
      "original_id": 101,
      "original_input": "The First Hospital Of Yulin",
      "state": "Shaanxi",
      "city": "Yulin"
    },
    {
      "original_id": 102,
      "original_input": "Yulin City first people's hospital",
      "state": "Guangxi Zhuang Autonomous Region",
      "city": "Yulin"
    },
    {
      "original_id": 103,
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
      "original_id": 101,
      "original_input": "The First Hospital Of Yulin",
      "state": "Shaanxi",
      "city": "Yulin",
      "original_input_zh": "榆林市第一医院"
    },
    {
      "original_id": 102,
      "original_input": "Yulin City first people's hospital",
      "state": "Guangxi Zhuang Autonomous Region",
      "city": "Yulin",
      "original_input_zh": "玉林市第一人民医院"
    },
    {
      "original_id": 103,
      "original_input": "Anfu Center Hospital",
      "state": "Jiangxi",
      "city": "Ji'an",
      "original_input_zh": "安福县中心医院 (推测)"
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

*   `"original_id"`: [Integer] - The unique numeric ID for the input record.
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
      "original_id": 12345,
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
      "original_id": 67890,
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

**# Confidence Level:**
- "高": Clear, unambiguous evidence (e.g., exact match, official abbreviation).
- "中": Strong but not conclusive evidence (e.g., common variation, no official confirmation).
- "低": Weak or ambiguous evidence (e.g., similar names, no clear web confirmation).

**# Output Format:**
```json
{
  "results": [
    {
      "original_id": 54321,
      "pi_site_name": "北医三院",
      "matched_site": "北京大学第三医院",
      "is_same": true,
      "confidence": "高",
      "reason": "名称包含关系，经搜索验证确认'北医三院'是'北京大学第三医院'的官方简称"
    }
  ]
}
```
"""
BATCH_ARBITRATE_PROMPT = """
**# Role**
You are a **Chief Medical Institution Adjudicator**. Your expertise lies in resolving complex discrepancies between institutional data points. Your task is to act as the final authority, making a definitive judgment based on rigorous, evidence-based investigation.

**# Core Principles**
1.  **Primacy of the Raw Input**: The `affiliation` (the raw input string) and its geographical context (`state`, `city`) are the ultimate source of truth. Your entire analysis must revolve around what *they* most likely represent.
2.  **Evidence over Proximity**: Do not simply choose the name that looks more similar. You must use web search to find evidence (e.g., official websites, historical names, addresses) to support your conclusion.
3.  **Acknowledge Ambiguity**: If the raw input is too ambiguous to definitively link to either candidate, you must state this.

**# Task: The Three-Way Arbitration**
For each input object, you are given:
- **The Raw Input**: `affiliation` (the original, often English, name) and its location (`state`, `city`).
- **Candidate A**: `pi_site_name` (the original human-provided Chinese label).
- **Candidate B**: `matched_site` (a machine-matched Chinese candidate).

You already know that **Candidate A and Candidate B are different entities**. Your task is to decide: **Does the Raw Input (`affiliation`) refer to Candidate A, Candidate B, or neither?**

**# Workflow: A Step-by-Step Arbitration Process**
1.  **Analyze the Raw Input**: What is the core entity described in the `affiliation`? What is its key geographical location from `state` and `city`?
2.  **Investigate Candidate A**: Perform a web search using `pi_site_name` and the location. Does the evidence (official English name, aliases, location) for `pi_site_name` strongly align with the Raw Input (`affiliation`)?
3.  **Investigate Candidate B**: Perform a web search using `matched_site` and the location. Does the evidence for `matched_site` strongly align with the Raw Input (`affiliation`)?
4.  **Compare and Conclude**: Weigh the evidence for both candidates.
    - If evidence for one is overwhelmingly stronger, choose that one.
    - If both are plausible but one is clearly better, choose the better one.
    - If the Raw Input is ambiguous or refers to a third, different entity, choose "Neither".

**# Output Format and Structure (JSON) - Must be strictly followed**
Your output must be a single JSON object containing a `results` list.

Each result object **must** contain the following keys:

*   `"original_id"`: [Integer] - Copied verbatim from the input.
*   `"decision"`: [String] - Your final verdict. Must be one of three exact values: **"pi_site_name"**, **"matched_site"**, or **"neither"**.
*   `"final_site"`: [String or null] - If your `decision` is "pi_site_name", this field must be the value of `pi_site_name`. If "matched_site", it must be the value of `matched_site`. If "neither", it must be `null`.
*   `"decision_reason"`: [String] - In Chinese. A detailed explanation of your arbitration process, comparing the evidence for both candidates and justifying your final `decision`.

**# Example**
**User Input:**
```json
{
  "inputs": [
    {
      "original_id": 123,
      "affiliation": "Cancer Hospital of CAMS",
      "state": "Beijing",
      "city": "Beijing",
      "pi_site_name": "中山大学肿瘤防治中心",
      "matched_site": "中国医学科学院肿瘤医院"
    }
  ]
}
```
Your Expected Output:
```json
{
  "results": [
    {
      "original_id": 123,
      "decision": "matched_site",
      "final_site": "中国医学科学院肿瘤医院",
      "decision_reason": "仲裁开始。原始输入'Cancer Hospital of CAMS'是'中国医学科学院肿瘤医院'的常见英文缩写。候选A'中山大学肿瘤防治中心'位于广州，与输入地点'Beijing'不符。候选B'中国医学科学院肿瘤医院'位于北京，且其英文名与输入高度匹配。证据明确指向候选B。因此，最终裁定原始输入指代的是'matched_site'。"
    }
  ]
}
```
"""

def is_chinese(text: str) -> bool:
    """Check if a string contains Chinese characters."""
    if not isinstance(text, str):
        return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def async_saver_local(output_filepath: str, local_queue: Queue):
    """Asynchronous saving function using a local queue."""
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
                if shutdown_event.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"异步保存失败: {str(e)}")

def get_tasks_to_process_generic(input_df: pd.DataFrame, id_column: str, output_jsonl_path: str, output_id_column: str = None) -> Tuple[List[Dict], pd.DataFrame]:
    """Calculate tasks to process, supporting checkpointing."""
    if output_id_column is None:
        output_id_column = id_column

    logger.info("--- 开始数据核验与差异计算 ---")
    
    if id_column not in input_df.columns:
        raise ValueError(f"错误: 输入DataFrame中未找到标识列 '{id_column}'。")
    
    input_df = input_df.drop_duplicates(subset=[id_column]).reset_index(drop=True)
    source_tasks_map = {row[id_column]: row.to_dict() for _, row in input_df.iterrows()}
    
    logger.info(f"从源数据加载了 {len(source_tasks_map)} 个独立任务。")

    processed_task_ids = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('status') == 'success' and output_id_column in data:
                        processed_task_ids.add(data[output_id_column])
                except json.JSONDecodeError:
                    logger.warning(f"无效JSON在 {output_jsonl_path} 第 {line_num} 行: {line[:100]}...")
        logger.info(f"从输出文件 '{os.path.basename(output_jsonl_path)}' 加载了 {len(processed_task_ids)} 条已处理记录。")
    else:
        logger.info(f"输出文件 '{os.path.basename(output_jsonl_path)}' 不存在，将处理所有任务。")
    
    tasks_to_run_ids = set(source_tasks_map.keys()) - processed_task_ids
    tasks_to_run = [source_tasks_map[task_id] for task_id in tasks_to_run_ids]
    
    logger.info(f"核验完成。总任务数: {len(source_tasks_map)}, 已处理: {len(processed_task_ids)}, 待处理: {len(tasks_to_run)}")
    return tasks_to_run, input_df

def cleanup_jsonl_file(filepath: str, id_column: str):
    """Clean a JSONL file, keeping only the latest successful records."""
    if not os.path.exists(filepath):
        return

    logger.info(f"--- 正在清理输出文件: {os.path.basename(filepath)} ---")
    
    all_records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    if id_column not in df.columns:
        logger.warning(f"找不到ID列 '{id_column}'，跳过清理。")
        return

    df_latest = df.drop_duplicates(subset=[id_column], keep='last')
    df_clean = df_latest[df_latest['status'] == 'success'].copy()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(df_clean.to_json(orient='records', lines=True, force_ascii=False))
    
    logger.info(f"清理完成。文件包含 {len(df_clean)} 条成功记录。")
