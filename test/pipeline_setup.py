import os
from crewai import Agent, LLM
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1. 配置 Hugging Face 访问令牌（若使用私有模型）
os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", "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")

# 2. CrewAI: 直接用 Transformers 加载 Mistral‐7B 模型
model_id_crew = "mradermacher/Mistral_7B_CrewAI-GGUF"
tokenizer_crew = AutoTokenizer.from_pretrained(model_id_crew)  # :contentReference[oaicite:0]{index=0}
model_crew     = AutoModel.from_pretrained(model_id_crew, torch_dtype="auto")  # :contentReference[oaicite:1]{index=1}

# 用 Transformers pipeline 包装为文本生成服务
crew_pipe = pipeline(
    "text-generation",
    model=model_crew,
    tokenizer=tokenizer_crew,
    device_map="auto"
)  # :contentReference[oaicite:2]{index=2}

# 将 pipeline 注入到 CrewAI 的 LLM 封装中
hf_llm_for_crewai = LLM(pipeline=crew_pipe)
agent = Agent(
    role="Carer Data Generator",
    goal="Generate carers data records",
    backstory="You generate synthetic Carers service records.",
    llm=hf_llm_for_crewai,
    verbose=True
)

# 3. LangChain: 加载 Llama‐2 7B LangChain Chat 模型
model_id_llama = "YanaS/llama-2-7b-langchain-chat-GGUF"
tokenizer_lc = AutoTokenizer.from_pretrained(model_id_llama)  # :contentReference[oaicite:3]{index=3}
model_lc = AutoModel.from_pretrained(model_id_llama, torch_dtype="auto")

# 高级 helper：Transformers pipeline
lc_pipe = pipeline(
    "text-generation",
    model=model_lc,
    tokenizer=tokenizer_lc,
    device_map="auto"
)

# 用 LangChain 封装
langchain_llm = HuggingFacePipeline(pipeline=lc_pipe)

# 导出供调用
__all__ = [
    "agent",
    "langchain_llm",
]



'''
# 1. 初始化 CrewAI Agent
#    使用从 Hugging Face Hub 拉取的 GGUF 模型
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
#os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# 2. 加载 tokenizer 和 model，禁用 fast tokenizer 转换
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise RuntimeError("请先设置环境变量 HUGGINGFACEHUB_API_TOKEN")

agent = Agent(
    role="Carer Data Generator",
    goal="Generate carers data records",
    backstory="You generate synthetic Carers service records.",
    llm="huggingface/EleutherAI/gpt-j-6B",
    verbose=True
)
#代码结构参考 crewAI-examples/starter_template :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

# 2. 初始化 LangChain 本地管道
#    这里以 Vicuna-13B LangChain 微调模型为例
hf_llm = HuggingFaceEndpoint(
    repo_id="EleutherAI/gpt-j-6b",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_length=128,
    provider="auto"  # 可选，默认也可省略
)

hf_llm = HuggingFaceHub(
    repo_id="EleutherAI/gpt-j-6b",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 128}
) # 参考 LangChain 本地管道示例 :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}



# 1. Mistral 7B Instruct (GPTQ) —— 可在本地量化后加载
mistral_pipeline = pipeline(
    task="text-generation",
    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. Llama-2-7B-Chat (HF) —— 官方 Chat 微调版
llama_pipeline = pipeline(
    task="text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)
'''