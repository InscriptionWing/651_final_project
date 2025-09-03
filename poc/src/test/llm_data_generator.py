"""
LLM驱动的护工数据生成器
使用大语言模型生成真实的护工服务记录
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import asdict

import openai
from faker import Faker
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from carer_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, DataValidator
)
from config import get_config, LLM_CONFIG, DATA_GENERATION_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker
fake = Faker(['en_AU', 'zh_CN'])


class LLMDataGenerator:
    """LLM驱动的护工数据生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.llm_config = self.config["llm"]
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 初始化LLM客户端
        self.client = self._init_llm_client()
        
        # 加载模板
        self.templates = self._load_templates()
        
        # 初始化数据验证器
        self.validator = DataValidator()
        
        # 预生成的护工和参与者档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info(f"LLM数据生成器初始化完成，使用模型: {self.llm_config['provider']}")
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        provider = self.llm_config["provider"]
        
        if provider == "openai":
            openai.api_key = self.llm_config["openai"]["api_key"]
            return openai
        elif provider == "anthropic":
            # 这里可以添加Anthropic客户端初始化
            logger.warning("Anthropic客户端尚未实现，回退到OpenAI")
            return openai
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    def _load_templates(self) -> List[str]:
        """加载增强模板"""
        template_file = Path("templates_enhanced.txt")
        if not template_file.exists():
            logger.warning(f"模板文件不存在: {template_file}")
            return []
        
        templates = []
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 解析模板格式：[outcome] narrative
            for line in content.strip().split('\n'):
                if line.strip() and '[' in line and ']' in line:
                    templates.append(line.strip())
        
        logger.info(f"加载了 {len(templates)} 个模板")
        return templates
    
    def generate_carer_profiles(self, count: int = 50) -> List[CarerProfile]:
        """生成护工档案"""
        profiles = []
        config = self.config["carer_profile"]
        
        for i in range(count):
            carer_id = f"CR{random.randint(100000, 999999):06d}"
            
            profile = CarerProfile(
                carer_id=carer_id,
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                certification_level=random.choice(config["certification_levels"]),
                years_experience=random.randint(*config["experience_range"]),
                specializations=random.sample(
                    config["specializations"], 
                    random.randint(1, 3)
                ),
                available_hours_per_week=random.randint(*config["hours_range"]),
                languages=random.sample(
                    config["languages"],
                    random.randint(1, 2)
                )
            )
            profiles.append(profile)
        
        logger.info(f"生成了 {len(profiles)} 个护工档案")
        return profiles
    
    def generate_participant_profiles(self, count: int = 100) -> List[ParticipantProfile]:
        """生成参与者档案"""
        profiles = []
        config = self.config["participant_profile"]
        
        for i in range(count):
            participant_id = f"PT{random.randint(100000, 999999):06d}"
            
            profile = ParticipantProfile(
                participant_id=participant_id,
                age_group=random.choice(config["age_groups"]),
                disability_type=random.choice(config["disability_types"]),
                support_level=random.choice(config["support_levels"]),
                communication_preferences=random.sample(
                    config["communication_preferences"],
                    random.randint(1, 2)
                ),
                mobility_requirements=random.choice([
                    [], ["wheelchair"], ["walking aid"], ["transfer assistance"]
                ])
            )
            profiles.append(profile)
        
        logger.info(f"生成了 {len(profiles)} 个参与者档案")
        return profiles
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM生成内容"""
        try:
            response = await self.client.ChatCompletion.acreate(
                model=self.llm_config["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm_config["openai"]["max_tokens"],
                temperature=self.llm_config["openai"]["temperature"],
                timeout=self.llm_config["openai"]["timeout"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise
    
    def _build_generation_prompt(self, 
                                service_type: ServiceType,
                                participant_profile: ParticipantProfile,
                                carer_profile: CarerProfile,
                                template: Optional[str] = None) -> str:
        """构建数据生成提示"""
        
        # 选择模板
        if template is None and self.templates:
            template = random.choice(self.templates)
        
        # 提取模板中的结果类型和叙述
        template_outcome = "positive"
        template_narrative = ""
        if template and '[' in template and ']' in template:
            match = re.match(r'\[(\w+)\]\s*(.*)', template)
            if match:
                template_outcome = match.group(1)
                template_narrative = match.group(2)
        
        prompt = f"""
作为NDIS护工服务记录专家，请生成一条真实的护工服务记录。

参与者信息：
- 年龄组: {participant_profile.age_group}
- 残疾类型: {participant_profile.disability_type}
- 支持级别: {participant_profile.support_level}
- 沟通偏好: {', '.join(participant_profile.communication_preferences)}

护工信息：
- 认证级别: {carer_profile.certification_level}
- 经验年限: {carer_profile.years_experience}年
- 专业领域: {', '.join(carer_profile.specializations)}

服务类型: {service_type.value}

参考模板（{template_outcome}结果）:
{template_narrative}

请生成一个JSON格式的服务记录，包含以下字段：
{{
  "narrative_notes": "详细的服务记录叙述（150-300字符，中文）",
  "service_outcome": "positive/neutral/negative/incomplete之一",
  "location_details": "具体服务地点描述",
  "support_techniques_used": ["使用的支持技术列表"],
  "challenges_encountered": ["遇到的挑战列表"],
  "participant_response": "参与者反应描述",
  "follow_up_required": true/false
}}

要求：
1. 叙述要具体、专业、符合NDIS标准
2. 体现护工的专业技能和参与者的个性化需求
3. 结果要与参考模板的情况类似
4. 所有内容必须是虚构的，不涉及真实个人信息
"""
        
        return prompt
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """生成单条服务记录"""
        
        try:
            # 构建提示
            prompt = self._build_generation_prompt(service_type, participant, carer)
            
            # 调用LLM
            llm_response = await self._call_llm(prompt)
            
            # 解析LLM响应
            llm_data = json.loads(llm_response)
            
            # 生成基础记录数据
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
            # 确定服务时长
            duration_ranges = self.config["service"]["duration_ranges"]
            duration_range = duration_ranges.get(service_type.value, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            # 确定地点
            location_weights = self.config["location"]["location_weights"]
            location_type = random.choices(
                list(LocationType),
                weights=[location_weights.get(lt.value, 0.01) for lt in LocationType]
            )[0]
            
            # 创建服务记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=llm_data.get("narrative_notes", ""),
                location_type=location_type,
                location_details=llm_data.get("location_details"),
                service_outcome=ServiceOutcome(llm_data.get("service_outcome", "positive")),
                support_techniques_used=llm_data.get("support_techniques_used", []),
                challenges_encountered=llm_data.get("challenges_encountered", []),
                participant_response=llm_data.get("participant_response"),
                follow_up_required=llm_data.get("follow_up_required", False)
            )
            
            # 验证记录
            errors = self.validator.validate_service_record(record)
            if errors:
                logger.warning(f"记录验证失败: {errors}")
                return None
            
            return record
            
        except Exception as e:
            logger.error(f"生成服务记录失败: {e}")
            return None
    
    async def generate_dataset(self, size: int = 1000) -> List[CarerServiceRecord]:
        """生成完整数据集"""
        logger.info(f"开始生成 {size} 条服务记录")
        
        # 生成档案
        self.carers = self.generate_carer_profiles(max(10, size // 20))
        self.participants = self.generate_participant_profiles(max(20, size // 10))
        
        records = []
        batch_size = self.gen_config["default_batch_size"]
        
        # 服务类型权重
        service_weights = self.config["service"]["service_types_weights"]
        service_types = list(ServiceType)
        weights = [service_weights.get(st.value, 0.1) for st in service_types]
        
        # 批量生成
        for batch_start in range(0, size, batch_size):
            batch_end = min(batch_start + batch_size, size)
            batch_tasks = []
            
            for i in range(batch_start, batch_end):
                # 随机选择护工和参与者
                carer = random.choice(self.carers)
                participant = random.choice(self.participants)
                
                # 生成服务日期（过去90天内）
                days_ago = random.randint(1, 90)
                service_date = date.today() - timedelta(days=days_ago)
                
                # 选择服务类型
                service_type = random.choices(service_types, weights=weights)[0]
                
                # 创建异步任务
                task = self.generate_service_record(
                    carer, participant, service_date, service_type
                )
                batch_tasks.append(task)
            
            # 执行批量任务
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 收集成功的记录
            for result in batch_results:
                if isinstance(result, CarerServiceRecord):
                    records.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"批量生成错误: {result}")
            
            logger.info(f"完成批次 {batch_start}-{batch_end}, 成功生成 {len([r for r in batch_results if isinstance(r, CarerServiceRecord)])} 条记录")
        
        logger.info(f"数据集生成完成，共 {len(records)} 条有效记录")
        return records
    
    def save_dataset(self, 
                    records: List[CarerServiceRecord], 
                    filename_prefix: str = "carers_data") -> Dict[str, str]:
        """保存数据集到多种格式"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config["output"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 转换为字典列表
        data_dicts = [record.to_dict() for record in records]
        
        # 保存JSON
        json_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, ensure_ascii=False, indent=2, default=str)
        saved_files["json"] = str(json_file)
        
        # 保存JSONL
        jsonl_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record_dict in data_dicts:
                f.write(json.dumps(record_dict, ensure_ascii=False, default=str) + '\n')
        saved_files["jsonl"] = str(jsonl_file)
        
        # 保存CSV
        try:
            import pandas as pd
            df = pd.DataFrame(data_dicts)
            csv_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            saved_files["csv"] = str(csv_file)
        except ImportError:
            logger.warning("pandas未安装，跳过CSV导出")
        
        logger.info(f"数据集已保存到: {saved_files}")
        return saved_files


async def main():
    """主函数 - 演示数据生成"""
    generator = LLMDataGenerator()
    
    # 生成小量数据用于测试
    test_size = 10
    logger.info(f"生成测试数据集（{test_size}条记录）")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records, "test_carers_data")
        
        # 打印质量报告
        quality_report = generator.validator.validate_data_quality(records)
        logger.info(f"数据质量报告: {quality_report}")
        
        # 显示示例记录
        if records:
            logger.info("示例记录:")
            print(records[0].to_json())
    else:
        logger.error("未能生成任何有效记录")


if __name__ == "__main__":
    asyncio.run(main())

