"""
免费LLM配置文件
包含各种免费LLM服务的配置选项
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# 项目基础配置
PROJECT_CONFIG = {
    "name": "NDIS_Carers_Data_Generator_Free",
    "version": "1.1.0",
    "description": "NDIS护工服务记录合成数据生成器 - 免费版本",
    "debug": True
}

# 免费LLM配置选项
FREE_LLM_CONFIG = {
    # 首选方法：Ollama本地模型（完全免费）
    "ollama": {
        "enabled": True,
        "base_url": "http://localhost:11434", #api: dc93835ea32d4a8bb097d80471b3f92c.i1gbA9QSQkpwu93m7d0dTop2
        "models": {
            "primary": "llama2",        # 推荐：轻量级，速度快
            "alternative": "mistral",   # 备选：质量更高
            "chinese": "qwen:7b"        # 中文支持更好
        },
        "generation_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500,
            "timeout": 30
        }
    },
    
    # 备选：Hugging Face免费API（需要注册免费账号）
    "huggingface": {
        "enabled": True,
        "token": "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",  # 需要用户替换
        "models": {
            "text_generation": "microsoft/DialoGPT-medium",
            "chinese_model": "THUDM/chatglm-6b",
            "medical_model": "microsoft/BioGPT"
        },
        "api_url_template": "https://api-inference.huggingface.co/models/{model}",
        "generation_params": {
            "max_length": 200,
            "temperature": 0.7,
            "do_sample": True,
            "timeout": 30
        }
    },
    
    # 本地运行的开源模型（需要GPU）
    "local_models": {
        "enabled": False,  # 默认关闭，需要用户手动启用
        "models": {
            "chinese_medical": {
                "name": "ChatGLM-6B",
                "path": "./models/chatglm-6b",
                "requirements": ["torch", "transformers", "sentencepiece"]
            },
            "general": {
                "name": "LLaMA-7B",
                "path": "./models/llama-7b",
                "requirements": ["torch", "transformers", "accelerate"]
            }
        }
    },
    
    # 基于模板的生成（无需LLM，质量很好）
    "template_based": {
        "enabled": True,
        "priority": 3,  # 优先级（数字越小优先级越高）
        "template_file": "templates_enhanced.txt",
        "custom_templates": {
            "positive_templates": [
                "为参与者{participant_name}提供{service_type}服务。参与者积极配合，使用{technique}方法取得良好效果。",
                "协助{participant_name}完成{service_type}活动。过程顺利，参与者表现出色，{technique}策略很有效。",
                "今日为{participant_name}进行{service_type}。参与者配合度高，通过{technique}达到预期目标。"
            ],
            "neutral_templates": [
                "为{participant_name}提供{service_type}服务。参与者表现平稳，采用{technique}方法按计划完成。",
                "协助{participant_name}进行{service_type}。过程正常，使用{technique}技术，效果一般。",
                "今日{participant_name}的{service_type}服务按计划进行，运用{technique}方法。"
            ],
            "negative_templates": [
                "为{participant_name}提供{service_type}服务时遇到挑战。尝试{technique}方法，需要调整策略。",
                "协助{participant_name}进行{service_type}遇到困难。使用{technique}技术缓解，效果有限。",
                "今日{participant_name}的{service_type}服务不够顺利，采用{technique}方法需要改进。"
            ]
        }
    },
    
    # 基于规则的生成（完全本地，速度最快）
    "rule_based": {
        "enabled": True,
        "priority": 4,
        "narrative_components": {
            "openings": [
                "为参与者提供专业的{service_type}服务，",
                "今日协助参与者进行{service_type}活动，",
                "在护工指导下，参与者参与{service_type}，",
                "根据参与者需求，实施{service_type}支持，"
            ],
            "process_descriptions": [
                "过程中采用{technique}方法，确保服务质量。",
                "运用{technique}策略，促进参与者积极参与。",
                "通过{technique}技术，提供个性化支持。",
                "实施{technique}方案，满足参与者需求。"
            ],
            "outcomes": {
                "positive": [
                    "参与者反应积极，达到预期目标。",
                    "活动进行顺利，参与者表现出色。",
                    "服务效果良好，参与者满意度高。"
                ],
                "neutral": [
                    "参与者表现平稳，完成基本目标。",
                    "活动按计划进行，无特殊情况。",
                    "服务过程正常，参与者配合一般。"
                ],
                "negative": [
                    "遇到一些挑战，需要后续跟进。",
                    "参与者情绪波动，需要额外支持。",
                    "活动未完全达到预期，需要调整。"
                ]
            }
        }
    }
}

# 免费API服务配置
FREE_API_SERVICES = {
    # Google AI Studio (Gemini) - 免费配额
    "google_ai": {
        "enabled": False,  # 需要用户配置
        "api_key": "your_google_ai_key_here",
        "model": "gemini-pro",
        "free_limit": "60 requests per minute"
    },
    
    # Cohere免费层
    "cohere": {
        "enabled": False,
        "api_key": "your_cohere_key_here", 
        "model": "command",
        "free_limit": "100 requests per month"
    },
    
    # Together AI免费配额
    "together_ai": {
        "enabled": False,
        "api_key": "your_together_key_here",
        "models": ["togethercomputer/llama-2-7b-chat"],
        "free_limit": "$25 monthly credit"
    }
}

# 数据生成配置（针对免费使用优化）
DATA_GENERATION_CONFIG = {
    "default_batch_size": 20,  # 减小批次大小，避免API限制
    "min_narrative_length": 50,
    "max_narrative_length": 300,  # 减少长度，提高生成速度
    "target_dataset_size": 500,   # 默认目标减少到500条
    "max_concurrent_requests": 2,  # 减少并发，避免触发限制
    "request_delay": 1.0,         # 请求间隔（秒）
    "retry_count": 2,             # 减少重试次数
    "random_seed": 42,
    "fallback_enabled": True      # 启用回退机制
}

# 模型自动检测和选择策略
AUTO_SELECTION_STRATEGY = {
    "priority_order": [
        "ollama",           # 首选：本地Ollama
        "template_based",   # 次选：基于模板
        "rule_based",       # 再次：基于规则
        "huggingface",      # 最后：免费API
    ],
    "fallback_enabled": True,
    "quality_threshold": 0.7,
    "speed_preference": "balanced"  # "fast", "balanced", "quality"
}

# 本地模型安装指南
OLLAMA_SETUP_GUIDE = {
    "installation_steps": [
        "1. 下载并安装Ollama: https://ollama.ai/download",
        "2. 启动Ollama服务",
        "3. 安装推荐模型: ollama pull llama2",
        "4. 可选安装中文模型: ollama pull qwen:7b",
        "5. 检查安装: ollama list"
    ],
    "recommended_models": {
        "llama2": {
            "size": "3.8GB",
            "description": "Meta的Llama 2模型，英文效果好",
            "command": "ollama pull llama2"
        },
        "mistral": {
            "size": "4.1GB", 
            "description": "Mistral AI模型，代码和推理能力强",
            "command": "ollama pull mistral"
        },
        "qwen:7b": {
            "size": "4.0GB",
            "description": "阿里通义千问，中文支持优秀",
            "command": "ollama pull qwen:7b"
        },
        "codellama": {
            "size": "3.8GB",
            "description": "专门用于代码生成的Llama版本",
            "command": "ollama pull codellama"
        }
    }
}

# Hugging Face设置指南
HUGGINGFACE_SETUP_GUIDE = {
    "steps": [
        "1. 访问 https://huggingface.co/ 注册免费账号",
        "2. 进入 Settings -> Access Tokens",
        "3. 创建新的 Read 权限 token",
        "4. 将token复制到配置文件中",
        "5. 免费账户每月有一定的API调用限制"
    ],
    "free_models": [
        "microsoft/DialoGPT-medium",
        "gpt2",
        "distilbert-base-uncased",
        "facebook/blenderbot-400M-distill"
    ]
}

def get_free_config() -> Dict[str, Any]:
    """获取免费LLM配置"""
    return {
        "project": PROJECT_CONFIG,
        "free_llm": FREE_LLM_CONFIG,
        "free_apis": FREE_API_SERVICES,
        "data_generation": DATA_GENERATION_CONFIG,
        "auto_selection": AUTO_SELECTION_STRATEGY,
        "setup_guides": {
            "ollama": OLLAMA_SETUP_GUIDE,
            "huggingface": HUGGINGFACE_SETUP_GUIDE
        }
    }

def check_available_services() -> Dict[str, bool]:
    """检查可用的免费LLM服务"""
    available = {}
    
    # 检查Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        available["ollama"] = response.status_code == 200
    except:
        available["ollama"] = False
    
    # 检查Hugging Face token
    hf_token = FREE_LLM_CONFIG["huggingface"]["token"]
    available["huggingface"] = hf_token and hf_token != "your_huggingface_token_here"
    
    # 本地方法总是可用
    available["template_based"] = True
    available["rule_based"] = True
    
    return available

def get_setup_instructions() -> str:
    """获取设置说明"""
    available = check_available_services()
    
    instructions = ["🚀 免费LLM数据生成器设置指南\n"]
    
    if not available["ollama"]:
        instructions.append("📥 推荐设置Ollama本地模型（完全免费，无限制）:")
        for step in OLLAMA_SETUP_GUIDE["installation_steps"]:
            instructions.append(f"   {step}")
        instructions.append("")
    
    if not available["huggingface"]:
        instructions.append("🤗 可选设置Hugging Face免费API:")
        for step in HUGGINGFACE_SETUP_GUIDE["steps"]:
            instructions.append(f"   {step}")
        instructions.append("")
    
    instructions.append("✅ 当前可用的生成方法:")
    for service, is_available in available.items():
        status = "🟢 可用" if is_available else "🔴 不可用"
        instructions.append(f"   {service}: {status}")
    
    if available["template_based"] or available["rule_based"]:
        instructions.append("\n💡 即使没有配置外部API，您也可以使用基于模板和规则的生成方法！")
    
    return "\n".join(instructions)

if __name__ == "__main__":
    # 显示设置指南
    print(get_setup_instructions())
    
    # 显示可用服务
    available = check_available_services()
    print(f"\n当前可用服务: {available}")

