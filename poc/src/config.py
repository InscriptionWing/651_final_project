"""
护工数据生成项目配置文件
包含所有项目设置和参数
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# 项目基础配置
PROJECT_CONFIG = {
    "name": "NDIS_Carers_Data_Generator",
    "version": "1.0.0",
    "description": "NDIS护工服务记录合成数据生成器",
    "debug": True
}

# LLM配置
LLM_CONFIG = {
    "provider": "openai",  # openai, anthropic, ollama
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", "your_openai_api_key_here"),
        "model": "gpt-3.5-turbo",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 30
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", "your_anthropic_key_here"),
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "timeout": 60
    }
}

# 数据生成配置
DATA_GENERATION_CONFIG = {
    "default_batch_size": 50,
    "min_narrative_length": 50,
    "max_narrative_length": 500,
    "target_dataset_size": 1000,
    "max_concurrent_requests": 5,
    "retry_count": 3,
    "random_seed": 42
}

# 输出配置
OUTPUT_CONFIG = {
    "output_dir": "./output",
    "formats": ["csv", "json", "jsonl"],
    "backup_enabled": True,
    "filename_template": "carers_data_{timestamp}_{size}records",
    "compression": False
}

# 数据质量和验证配置
VALIDATION_CONFIG = {
    "validation_enabled": True,
    "privacy_check_enabled": True,
    "quality_threshold": 0.85,
    "duplicate_check": True,
    "schema_validation": True,
    "content_validation": True
}

# 模板和训练数据配置
TEMPLATE_CONFIG = {
    "template_file": "./templates_enhanced.txt",
    "custom_templates_dir": "./templates",
    "use_enhanced_templates": True,
    "template_categories": ["positive", "neutral", "negative"]
}

# 护工档案生成配置
CARER_PROFILE_CONFIG = {
    "certification_levels": ["Certificate III", "Certificate IV", "Diploma", "Degree"],
    "specializations": [
        "个人护理", "行为支持", "认知支持", "身体残疾支持", 
        "心理健康支持", "老年护理", "儿童发展支持"
    ],
    "languages": ["English", "Mandarin", "Spanish", "Arabic", "Vietnamese", "Greek"],
    "experience_range": (0, 25),
    "hours_range": (10, 40)
}

# 参与者档案配置
PARTICIPANT_PROFILE_CONFIG = {
    "age_groups": ["18-25", "26-35", "36-50", "51-65", "65+"],
    "disability_types": [
        "智力残疾", "自闭症谱系障碍", "身体残疾", "感官残疾", 
        "心理社会残疾", "神经系统残疾", "多重残疾"
    ],
    "support_levels": ["Low", "Medium", "High", "Complex"],
    "communication_preferences": [
        "口语交流", "手语", "图片交流", "文字交流", "辅助技术"
    ]
}

# 服务配置
SERVICE_CONFIG = {
    "service_types_weights": {
        "个人护理": 0.25,
        "家务支持": 0.15, 
        "社区参与": 0.20,
        "交通协助": 0.10,
        "社交支持": 0.15,
        "物理治疗": 0.05,
        "用药支持": 0.05,
        "技能发展": 0.05
    },
    "duration_ranges": {
        "个人护理": (0.5, 4.0),
        "家务支持": (1.0, 6.0),
        "社区参与": (1.0, 8.0),
        "交通协助": (0.5, 3.0),
        "社交支持": (1.0, 4.0),
        "物理治疗": (0.5, 2.0),
        "用药支持": (0.25, 1.0),
        "技能发展": (1.0, 6.0)
    },
    "outcome_weights": {
        "positive": 0.60,
        "neutral": 0.25,
        "negative": 0.10,
        "incomplete": 0.05
    }
}

# 地点配置
LOCATION_CONFIG = {
    "location_weights": {
        "参与者家中": 0.45,
        "社区中心": 0.15,
        "医疗机构": 0.10,
        "购物中心": 0.10,
        "图书馆": 0.05,
        "游泳馆": 0.05,
        "药房": 0.05,
        "公园": 0.03,
        "其他": 0.02
    }
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "file": "./logs/generator.log",
    "rotation": "daily",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_size": "100MB",
    "backup_count": 7
}

# 隐私和安全配置
PRIVACY_CONFIG = {
    "anonymization_enabled": True,
    "secure_output": True,
    "data_retention_days": 365,
    "pii_detection": True,
    "encryption_enabled": False  # 可根据需要启用
}

# 性能配置
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size": 1000,
    "parallel_processing": True,
    "memory_limit": "2GB",
    "progress_reporting": True
}

def get_config() -> Dict[str, Any]:
    """获取完整配置字典"""
    return {
        "project": PROJECT_CONFIG,
        "llm": LLM_CONFIG,
        "data_generation": DATA_GENERATION_CONFIG,
        "output": OUTPUT_CONFIG,
        "validation": VALIDATION_CONFIG,
        "templates": TEMPLATE_CONFIG,
        "carer_profile": CARER_PROFILE_CONFIG,
        "participant_profile": PARTICIPANT_PROFILE_CONFIG,
        "service": SERVICE_CONFIG,
        "location": LOCATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "privacy": PRIVACY_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }

def get_output_directory() -> Path:
    """获取输出目录路径"""
    output_dir = Path(OUTPUT_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_logs_directory() -> Path:
    """获取日志目录路径"""
    log_file = Path(LOGGING_CONFIG["file"])
    log_dir = log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print("项目配置:")
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in settings.items():
            print(f"  {key}: {value}")

