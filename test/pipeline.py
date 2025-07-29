# -*- coding: utf-8 -*-
"""
端到端合成流水线示例，使用 LangGraphClient 和 CrewAIClient 的原生接口
依赖：langgraph, crewai, pandas, tqdm, jsonschema
"""

import json
import pandas as pd
from tqdm import tqdm
from langgraph import LangGraphClient
from crewai import CrewAIClient
import jsonschema

# ———— 1. 客户端初始化 ————
lg_client = LangGraphClient(
    api_key="YOUR_LANGGRAPH_API_KEY",
    model="gpt-4o"               # 或者你在 LangGraph 控制台里选的模型
)
cr_client = CrewAIClient(
    api_key="YOUR_CREWAI_API_KEY",
    default_model="gpt-4o"      # CrewAI 后端调用的 LLM
)

# ———— 2. Schema 校验工具 ————
with open("schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def validate_and_fill(record: dict) -> dict:
    """用 jsonschema 校验，并填充缺失默认值（可选）"""
    jsonschema.validate(instance=record, schema=schema)
    # 这里可以做填 default 值的工作
    return record

# ———— 3. Pipeline 各环节原生调用 ————
def extract_graph(description: str) -> dict:
    """调用 LangGraph 构建因果图"""
    resp = lg_client.run_graph(
        input_text=description,
        graph_type="causal"   # 假定支持因果图
    )
    return resp["graph"]

def generate_synthetic(record: dict, graph: dict) -> dict:
    """调用 CrewAI 生成合成记录"""
    prompt = {
        "instruction": "请根据以下原始事故记录和因果图，生成一条新的、结构一致且 narrative 自然的合成记录。",
        "original_record": record,
        "event_graph": graph
    }
    resp = cr_client.invoke(
        agent_name="default-generator",  # 在 CrewAI 控制台定义过的 Agent 名称
        prompt=prompt
    )
    return resp["output"]["synthetic_record"]

def analyze_metrics(original: dict, synthetic: dict) -> dict:
    """调用 CrewAI 或自定义计算统计 & 隐私指标"""
    prompt = {
        "instruction": "请计算以下指标：KS 距离、JSD、MIA 成功率。",
        "original_record": original,
        "synthetic_record": synthetic
    }
    resp = cr_client.invoke(
        agent_name="default-analyzer",
        prompt=prompt
    )
    return resp["output"]["metrics"]

def report_comparison(original: dict, synthetic: dict, metrics: dict) -> dict:
    """调用 CrewAI 生成对比报告"""
    prompt = {
        "instruction": "请生成一份 JSON 格式的对比报告，包含原始 vs 合成摘要和各项指标解释。",
        "original_record": original,
        "synthetic_record": synthetic,
        "metrics": metrics
    }
    resp = cr_client.invoke(
        agent_name="default-reporter",
        prompt=prompt
    )
    return resp["output"]["report"]

# ———— 4. 主流程 ————
def load_originals(path: str) -> list:
    df = pd.read_json(path, orient="records")
    return [validate_and_fill(r) for r in df.to_dict(orient="records")]

def pipeline(record: dict) -> dict:
    graph = extract_graph(record["incident_description"])
    synthetic = generate_synthetic(record, graph)
    metrics = analyze_metrics(record, synthetic)
    report = report_comparison(record, synthetic, metrics)
    return {
        "synthetic": synthetic,
        "metrics": metrics,
        "report": report
    }

if __name__ == "__main__":
    originals = load_originals("data/original_incidents_100.json")
    all_results = []

    for rec in tqdm(originals, desc="生成合成数据"):
        result = pipeline(rec)
        # 合并到一条记录里
        combined = {**result["synthetic"], **{"metrics": result["metrics"], "report": result["report"]}}
        all_results.append(combined)

    # 保存合成数据集
    df_out = pd.json_normalize(all_results)
    df_out.to_csv("data/synthetic_incidents_100.csv", index=False, encoding="utf-8-sig")

    # 保存分析报告
    with open("data/analysis_reports_100.json", "w", encoding="utf-8") as f:
        json.dump([r["report"] for r in all_results], f, ensure_ascii=False, indent=2)

    print("✅ 已生成 synthetic_incidents_100.csv 和 analysis_reports_100.json")
