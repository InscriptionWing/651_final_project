#!/usr/bin/env python
import os
import json
import typer
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich import print
from pathlib import Path

load_dotenv()
app = typer.Typer(add_completion=False)

# 选择一个指令式大模型（示例：Falcon 7B Instruct）
MODEL_ID = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")  # 可替换为任何 text-generation 模型 :contentReference[oaicite:2]{index=2}

# 加载 tokenizer 与模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",        # 自动分配 GPU/CPU
    offload_folder="offload", # 大模型时可选
    torch_dtype="auto",
)

# 文本生成 pipeline，自动处理预/后处理
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    do_sample=True,
    temperature=0.7,
)

# 读取 Prompt 模板
TEMPLATE = Path(__file__).with_name("prompt_template.md").read_text()

@app.command()
def generate(
    activity: str = "reading session",
    location: str = "public library"
):
    """生成一条 incident JSON（仅输出 JSON）"""
    prompt = TEMPLATE.format(activity=activity, location=location)
    outputs = generator(prompt, return_full_text=False, num_return_sequences=1)
    # pipeline 返回列表，每项为 {"generated_text": "..."}
    raw = outputs[0]["generated_text"].strip()
    # 确保输出为纯 JSON
    try:
        obj = json.loads(raw)
        print(obj)
        return obj
    except json.JSONDecodeError:
        print("[bold red]✗ 解析失败，请检查模型输出是否为有效 JSON[/]")
        print(raw)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
