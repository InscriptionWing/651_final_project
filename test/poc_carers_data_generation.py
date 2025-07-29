
import os
import json
import csv
import uuid
from datetime import datetime
from langchain import PromptTemplate, LLMChain
#from langchain.llms import HuggingFaceHub

import os
# ─── Disable FP8 quantization on CPU ────────────────────────────────────────
os.environ["TRANSFORMERS_NO_FP8"] = "1"

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain

# 1. 安装依赖（首次运行时取消注释）
# !pip install transformers torch langchain

# 2. 初始化 Hugging Face 文本生成管道
model_id = "moonshotai/Kimi-K2-Instruct"  # POC 演示，可替换成更强模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    batch_size=4,
)

tokenizer.pad_token = tokenizer.eos_token  # 或者添加新的专用 pad_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

# 3. 创建 Hugging Face 文本生成 pipeline 并包装为 LangChain LLM
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.6,       # 推荐温度
    batch_size=4,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 4. 定义提示词模板
# 3. 定义转义后 f-string 模板
template = """
You are to generate {num_records} synthetic Carers data in JSON array format:
[
  {{
    "CarerID": "C{{i:04d}}",
    "ParticipantID": "P{{i:04d}}",
    "ServiceDate": "2025-07-22",
    "ServiceType": "HomeCare",
    "Duration": 2.5,
    "NarrativeNotes": "Routine check and assistance."
  }}
]
"""
prompt = PromptTemplate.from_template(template=template)

# 5. 构建并运行 LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.invoke({"num_records": 100})

# 6. 打印或保存结果
print(output)
# 可选：将 output 写入文件
with open("carers_full.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(output, ensure_ascii=False, indent=2))



'''
# 1. 环境配置：设置 HuggingFace Hub Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 2. LLM 初始化：选择 CrewAI/LangChain 驱动的模型
llm = HuggingFaceHub(
    repo_id="rinna/vicuna-13b-delta-finetuned-langchain-MRKL",  # 或者其他 CrewAI 优化模型
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# 3. Prompt 模板：生成 100 条 JSON 格式的 Carers 数据
prompt = PromptTemplate(
    input_variables=["n"],
    template="""
You are a synthetic data generator. Produce exactly {n} records in JSON array format.
Each record must include the following fields:
- CarerID: a UUID string
- ParticipantID: a UUID string
- ServiceDate: a random date in 2023 (YYYY-MM-DD)
- ServiceType: one of ["HomeVisit", "Respite", "TherapySession", "GroupActivity"]
- Duration: integer number of hours (1–8)
- NarrativeNotes: 1–2 sentence description of the service provided.

Return only the JSON array.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_records(n: int = 100) -> list:
    """调用 LLM，生成并解析 n 条记录"""
    print(f"Generating {n} records via LLM...")
    raw = chain.run({"n": n})
    # 有时模型会多输出文本，需要提取第一个 JSON 数组
    start = raw.find("[")
    end = raw.rfind("]") + 1
    json_str = raw[start:end]
    records = json.loads(json_str)
    return records

def export_json(records: list, path: str = "carers_poc.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Exported JSON → {path}")

def export_csv(records: list, path: str = "carers_poc.csv"):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported CSV  → {path}")

if __name__ == "__main__":
    # POC 实现：生成 & 导出
    carers_data = generate_records(n=100)
    export_json(carers_data, "poc_carers_100.json")
    export_csv(carers_data, "poc_carers_100.csv")
'''