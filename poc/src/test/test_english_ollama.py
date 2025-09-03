"""
英文版Ollama数据生成测试
专门为全英文输出优化
"""

import asyncio
import json
import logging
from datetime import date, timedelta
from typing import Optional
import random

from english_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, EnglishDataValidator
)
from faker import Faker
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化英文Faker
fake = Faker(['en_AU'])


class EnglishOllamaGenerator:
    """英文版Ollama数据生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.validator = EnglishDataValidator()
        
        # 检测可用的Ollama模型
        self.available_model = self._detect_ollama_model()
        if not self.available_model:
            raise Exception("No Ollama model available")
        
        logger.info(f"English Ollama Generator initialized with model: {self.available_model}")
    
    def _detect_ollama_model(self) -> Optional[str]:
        """检测可用的Ollama模型"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    return models[0]['name']
        except Exception as e:
            logger.error(f"Failed to detect Ollama model: {e}")
        return None
    
    async def _call_ollama(self, prompt: str) -> str:
        """调用Ollama生成英文内容"""
        try:
            data = {
                "model": self.available_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    def _build_english_prompt(self, service_type: ServiceType, participant_name: str, carer_name: str) -> str:
        """构建英文生成提示"""
        prompt = f"""Generate a professional NDIS carer service record narrative in English.

Service Details:
- Service Type: {service_type.value}
- Participant: {participant_name}
- Carer: {carer_name}
- Date: {date.today().strftime('%B %d, %Y')}

Requirements:
1. Write a detailed service narrative (100-250 words)
2. Include specific support techniques used
3. Describe participant response and engagement
4. Note any challenges or outcomes
5. Use professional, respectful language
6. Focus on person-centered care approach

Generate ONLY the narrative text, no JSON or formatting."""
        
        return prompt
    
    async def generate_english_record(self) -> Optional[CarerServiceRecord]:
        """生成单条英文服务记录"""
        
        try:
            # 生成基础数据
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            carer_id = f"CR{random.randint(100000, 999999):06d}"
            participant_id = f"PT{random.randint(100000, 999999):06d}"
            
            # 随机选择服务类型
            service_type = random.choice(list(ServiceType))
            
            # 生成日期（过去30天内）
            days_ago = random.randint(1, 30)
            service_date = date.today() - timedelta(days=days_ago)
            
            # 生成持续时间
            duration_ranges = {
                ServiceType.PERSONAL_CARE: (0.5, 4.0),
                ServiceType.HOUSEHOLD_TASKS: (1.0, 6.0),
                ServiceType.COMMUNITY_ACCESS: (2.0, 8.0),
                ServiceType.TRANSPORT: (0.5, 3.0),
                ServiceType.SOCIAL_SUPPORT: (1.0, 4.0),
                ServiceType.PHYSIOTHERAPY: (0.5, 2.0),
                ServiceType.MEDICATION_SUPPORT: (0.25, 1.0),
                ServiceType.SKILL_DEVELOPMENT: (1.0, 6.0),
                ServiceType.RESPITE_CARE: (2.0, 8.0),
                ServiceType.MEAL_PREPARATION: (0.5, 2.0)
            }
            duration_range = duration_ranges.get(service_type, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            # 生成名称
            participant_name = fake.first_name()
            carer_name = fake.name()
            
            # 使用Ollama生成英文叙述
            prompt = self._build_english_prompt(service_type, participant_name, carer_name)
            logger.info(f"Generating narrative for {service_type.value} service...")
            
            narrative = await self._call_ollama(prompt)
            
            # 确保叙述长度合适
            if len(narrative) < 80:
                narrative += f" The {service_type.value.lower()} session was conducted according to NDIS standards with focus on participant autonomy and dignity."
            elif len(narrative) > 500:
                narrative = narrative[:497] + "..."
            
            # 确定服务结果
            if "excellent" in narrative.lower() or "outstanding" in narrative.lower() or "successfully" in narrative.lower():
                service_outcome = ServiceOutcome.POSITIVE
            elif "challenging" in narrative.lower() or "difficult" in narrative.lower() or "limited" in narrative.lower():
                service_outcome = ServiceOutcome.NEGATIVE
            elif "incomplete" in narrative.lower() or "interrupted" in narrative.lower():
                service_outcome = ServiceOutcome.INCOMPLETE
            else:
                service_outcome = ServiceOutcome.NEUTRAL
            
            # 生成其他字段
            location_type = random.choice(list(LocationType))
            
            support_techniques = random.sample([
                "Person-centered communication", "Behavioral reinforcement", "Environmental modification",
                "Visual prompting", "Task breakdown", "Positive encouragement", "Structured guidance",
                "Adaptive equipment use", "Sensory regulation", "Routine establishment"
            ], random.randint(2, 4))
            
            challenges = []
            if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
                challenges = random.sample([
                    "Communication barriers", "Behavioral escalation", "Environmental sensitivities",
                    "Attention difficulties", "Physical limitations", "Equipment issues"
                ], random.randint(1, 2))
            
            participant_responses = {
                ServiceOutcome.POSITIVE: ["Highly engaged", "Very cooperative", "Enthusiastic participation"],
                ServiceOutcome.NEUTRAL: ["Cooperative", "Stable engagement", "Standard participation"],
                ServiceOutcome.NEGATIVE: ["Required encouragement", "Challenging session", "Limited cooperation"],
                ServiceOutcome.INCOMPLETE: ["Session interrupted", "Early completion", "Requires follow-up"]
            }
            
            # 创建记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer_id,
                carer_name=carer_name,
                participant_id=participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=narrative,
                location_type=location_type,
                location_details=f"{location_type.value} - Professional support environment",
                service_outcome=service_outcome,
                support_techniques_used=support_techniques,
                challenges_encountered=challenges,
                participant_response=random.choice(participant_responses.get(service_outcome, ["Standard response"])),
                follow_up_required=service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE],
                billing_code=f"NDIS_{service_type.value.replace(' ', '_').upper()}_{random.randint(1000, 9999)}",
                supervision_notes="Supervision completed as per NDIS requirements" if random.random() < 0.3 else None
            )
            
            # 验证记录
            errors = self.validator.validate_service_record(record)
            if errors:
                logger.warning(f"Record validation failed: {errors}")
                return None
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to generate English record: {e}")
            return None
    
    async def generate_test_dataset(self, size: int = 5) -> list:
        """生成测试数据集"""
        logger.info(f"Generating {size} English service records using Ollama...")
        
        records = []
        for i in range(size):
            try:
                record = await self.generate_english_record()
                if record:
                    records.append(record)
                    logger.info(f"Generated record {i+1}/{size}: {record.service_type.value}")
                else:
                    logger.warning(f"Failed to generate record {i+1}")
            except Exception as e:
                logger.error(f"Error generating record {i+1}: {e}")
        
        logger.info(f"Successfully generated {len(records)} records")
        return records
    
    def save_records(self, records: list, filename: str = "english_ollama_test"):
        """保存记录到JSON文件"""
        timestamp = date.today().strftime("%Y%m%d")
        filename_full = f"{filename}_{timestamp}_{len(records)}records.json"
        
        data = [record.to_dict() for record in records]
        
        with open(filename_full, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Records saved to: {filename_full}")
        return filename_full


async def main():
    """主测试函数"""
    try:
        generator = EnglishOllamaGenerator()
        
        # 生成测试数据
        records = await generator.generate_test_dataset(3)
        
        if records:
            # 保存数据
            saved_file = generator.save_records(records)
            
            # 显示示例
            print(f"\n✅ English Ollama test completed successfully!")
            print(f"📊 Generated {len(records)} records")
            print(f"📁 Saved to: {saved_file}")
            
            print(f"\n📋 Sample record:")
            sample_record = records[0]
            print(f"Service Type: {sample_record.service_type.value}")
            print(f"Duration: {sample_record.duration_hours} hours")
            print(f"Outcome: {sample_record.service_outcome.value}")
            print(f"Narrative: {sample_record.narrative_notes[:200]}...")
            
        else:
            print("❌ Failed to generate any records")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
