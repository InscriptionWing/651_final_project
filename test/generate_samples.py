import json
from pipeline_setup import agent, langchain_llm #hf_llm
from prompt_design import prompt

def generate_with_crewai(n=100):
    samples = []
    for _ in range(n):
        raw = agent.kickoff(prompt.format_prompt().to_string())
        samples.append(json.loads(raw))
    return samples

def generate_with_langchain(n=100):
    samples = []
    for _ in range(n):
        out = langchain_llm.generate(prompt.format_prompt().to_string())
        samples.append(json.loads(out.text))
    return samples

if __name__ == "__main__":
    crewai_samples   = generate_with_crewai(100)
    langchain_samples = generate_with_langchain(100)

    with open("crewai_samples.json", "w", encoding="utf-8") as f:
        json.dump(crewai_samples, f, ensure_ascii=False, indent=2)
    with open("langchain_samples.json", "w", encoding="utf-8") as f:
        json.dump(langchain_samples, f, ensure_ascii=False, indent=2)

    print("示例数据生成完毕，各 100 条。")
'''
def generate_with_crewai(n):
    samples = []
    for _ in range(n):
        # 用 kickoff() 同步生成
        result = agent.kickoff(prompt.format_prompt().to_string())
        # 从 LiteAgentOutput.raw 取原始字符串，再转 JSON
        samples.append(json.loads(result.raw))
    return samples

def generate_with_langchain(n):
    samples = []
    for _ in range(n):
        # LangChainHub 或 HuggingFaceHub 均提供 .run 或 .generate 方法
        # 假设 hf_llm 是 HuggingFaceHub 实例，使用 .run()
        raw = hf_llm.run(prompt.format_prompt().to_string())
        samples.append(json.loads(raw))
    return samples

if __name__ == "__main__":
    # 同步批量生成
    crewai_samples = generate_with_crewai(100)
    langchain_samples = generate_with_langchain(100)

    # 保存示例数据
    with open("crewai_samples.json", "w", encoding="utf-8") as f:
        json.dump(crewai_samples, f, ensure_ascii=False, indent=2)
    with open("langchain_samples.json", "w", encoding="utf-8") as f:
        json.dump(langchain_samples, f, ensure_ascii=False, indent=2)

    print("示例数据生成完毕，各 100 条。")  # 测试反馈，可用于基准对比 :contentReference[oaicite:10]{index=10}

    

    def generate_with_pipeline(gen_pipeline, n=100):
    samples = []
    for _ in range(n):
        # 生成
        out = gen_pipeline(
            PROMPT_TEMPLATE,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            eos_token_id=gen_pipeline.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        # 去除 prompt 前缀
        json_part = text[len(PROMPT_TEMPLATE):].strip()
        samples.append(json.loads(json_part))
    return samples

if __name__ == "__main__":
    # 各生成 100 条示例
    mistral_samples = generate_with_pipeline(mistral_pipeline, 100)
    llama_samples = generate_with_pipeline(llama_pipeline, 100)

    # 写入文件
    with open("mistral_samples.json", "w", encoding="utf-8") as f:
        json.dump(mistral_samples, f, ensure_ascii=False, indent=2)
    with open("llama_samples.json", "w", encoding="utf-8") as f:
        json.dump(llama_samples, f, ensure_ascii=False, indent=2)

    print("示例数据已生成：mistral_samples.json, llama_samples.json")
    
'''