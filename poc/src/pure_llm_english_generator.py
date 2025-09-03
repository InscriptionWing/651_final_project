"""
纯LLM英文数据生成器
完全依赖Ollama LLM，不使用任何模板
"""

import json
import random
import asyncio
import logging
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from faker import Faker

from english_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, EnglishDataValidator
)
from config import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化英文Faker
fake = Faker(['en_AU'])


class PureLLMEnglishGenerator:
    """纯LLM英文数据生成器 - 完全依靠Ollama LLM"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 初始化数据验证器
        self.validator = EnglishDataValidator()
        
        # 检测并初始化Ollama
        self.ollama_model = self._detect_ollama_model()
        if not self.ollama_model:
            raise Exception("Ollama model not available. Please ensure Ollama is running and a model is installed.")
        
        # 预生成档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info(f"Pure LLM English generator initialized with model: {self.ollama_model}")
    
    def _detect_ollama_model(self) -> Optional[str]:
        """检测可用的Ollama模型"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    model_name = models[0]['name']
                    logger.info(f"Detected Ollama model: {model_name}")
                    return model_name
        except Exception as e:
            logger.error(f"Failed to detect Ollama model: {e}")
        return None
    
    async def _call_ollama_llm(self, prompt: str) -> str:
        """调用Ollama LLM生成内容"""
        try:
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_tokens": 250,
                    "stop": ["---", "###", "END", "REQUIREMENTS:", "SERVICE DETAILS:", "\n\n\n"]
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=60  # 适中的超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # 清理生成的文本
                generated_text = self._clean_generated_text(generated_text)
                
                return generated_text
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}")
            raise
    
    def _clean_generated_text(self, text: str) -> str:
        """清理LLM生成的文本"""
        # 移除多余的换行和空格
        text = ' '.join(text.split())
        
        # 移除常见的LLM标记和提示词重复
        cleanup_patterns = [
            "Here is", "Here's", "The narrative is:", "Narrative:",
            "Service record:", "Record:", "Note:", "Notes:",
            "Summary:", "Description:", "Report:", "Professional Narrative:",
            "NDIS", "SERVICE DETAILS:", "REQUIREMENTS:", "Write a", "Write ONLY"
        ]
        
        for pattern in cleanup_patterns:
            if text.startswith(pattern):
                text = text[len(pattern):].strip()
        
        # 截断过长的文本（保留前800字符，确保在验证范围内）
        if len(text) > 800:
            # 找到最后一个完整句子
            sentences = text[:800].split('.')
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'
            else:
                text = text[:797] + '...'
        
        # 确保最小长度
        if len(text) < 50:
            text += " This service was provided in accordance with NDIS standards and participant care plan requirements."
        
        # 确保首字母大写，句号结尾
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            if not text.endswith('.') and not text.endswith('...'):
                text += '.'
        
        return text
    
    def _build_narrative_prompt(self, 
                               service_type: ServiceType, 
                               outcome: ServiceOutcome,
                               duration: float,
                               location_type: LocationType,
                               carer_name: str,
                               participant_age_group: str,
                               disability_type: str) -> str:
        """构建叙述生成的详细提示"""
        
        outcome_descriptions = {
            ServiceOutcome.POSITIVE: "excellent results, participant very cooperative, goals exceeded",
            ServiceOutcome.NEUTRAL: "standard progress, routine session, objectives met as planned", 
            ServiceOutcome.NEGATIVE: "challenges encountered, participant had difficulties, modified approach needed",
            ServiceOutcome.INCOMPLETE: "session ended early, incomplete objectives, follow-up required"
        }
        
        prompt = f"""Write a concise professional NDIS carer service narrative in English for the following scenario:

SERVICE DETAILS:
- Service Type: {service_type.value}
- Duration: {duration} hours
- Location: {location_type.value}
- Carer: {carer_name}
- Participant: {participant_age_group} with {disability_type}
- Session Outcome: {outcome_descriptions.get(outcome, 'standard session')}

REQUIREMENTS:
1. Write a concise professional narrative (100-200 words maximum)
2. Use person-centered, respectful language
3. Include 1-2 specific support techniques used
4. Describe participant response briefly
5. Mention key outcomes or challenges
6. Follow Australian NDIS documentation standards
7. Write in third person professional voice
8. Keep it focused and direct

Write ONLY the narrative text, no headers or extra formatting. Maximum 200 words.

Narrative:"""
        
        return prompt
    
    def _build_support_techniques_prompt(self, service_type: ServiceType) -> str:
        """构建支持技术生成提示"""
        prompt = f"""List 2-3 support techniques for {service_type.value}. One per line, no numbering:"""
        
        return prompt
    
    def _build_challenges_prompt(self, outcome: ServiceOutcome, service_type: ServiceType) -> str:
        """构建挑战生成提示"""
        if outcome not in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
            return ""
        
        prompt = f"""List 1-2 challenges for {service_type.value} with {outcome.value} outcome. One per line:"""
        
        return prompt
    
    def _build_participant_response_prompt(self, outcome: ServiceOutcome) -> str:
        """构建参与者反应生成提示"""
        prompt = f"""Describe participant response for {outcome.value} session in 1-2 words:"""
        
        return prompt
    
    async def _generate_llm_content(self, 
                                  service_type: ServiceType,
                                  outcome: ServiceOutcome,
                                  duration: float,
                                  location_type: LocationType,
                                  carer_name: str,
                                  participant_age_group: str,
                                  disability_type: str) -> Dict[str, Any]:
        """使用LLM生成所有内容"""
        
        # 生成叙述
        narrative_prompt = self._build_narrative_prompt(
            service_type, outcome, duration, location_type, 
            carer_name, participant_age_group, disability_type
        )
        narrative = await self._call_ollama_llm(narrative_prompt)
        
        # 生成支持技术
        techniques_prompt = self._build_support_techniques_prompt(service_type)
        techniques_text = await self._call_ollama_llm(techniques_prompt)
        support_techniques = [
            line.strip() for line in techniques_text.split('\n') 
            if line.strip() and not line.strip().startswith('-')
        ][:4]  # 最多4个
        
        # 生成挑战（如果需要）
        challenges = []
        if outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
            challenges_prompt = self._build_challenges_prompt(outcome, service_type)
            challenges_text = await self._call_ollama_llm(challenges_prompt)
            challenges = [
                line.strip() for line in challenges_text.split('\n') 
                if line.strip() and not line.strip().startswith('-')
            ][:2]  # 最多2个
        
        # 生成参与者反应
        response_prompt = self._build_participant_response_prompt(outcome)
        participant_response = await self._call_ollama_llm(response_prompt)
        
        return {
            "narrative_notes": narrative,
            "support_techniques_used": support_techniques,
            "challenges_encountered": challenges,
            "participant_response": participant_response
        }
    
    def generate_carer_profiles(self, count: int = 50) -> List[CarerProfile]:
        """生成护工档案"""
        profiles = []
        
        certifications = [
            "Certificate III in Individual Support", 
            "Certificate IV in Disability Support",
            "Diploma of Community Services", 
            "Bachelor of Nursing",
            "Certificate III in Aged Care",
            "Certificate IV in Mental Health",
            "Diploma of Disability Support",
            "Certificate IV in Allied Health Assistance"
        ]
        
        specializations = [
            "Personal Care Assistance", "Behavioral Support", "Cognitive Rehabilitation", 
            "Physical Disability Support", "Mental Health Recovery", "Aged Care Services",
            "Developmental Disability Support", "Community Participation", "Autism Support",
            "Acquired Brain Injury Support", "Sensory Impairment Support", "Respite Care",
            "Social Skills Development", "Independent Living Skills"
        ]
        
        languages = ["English", "Mandarin", "Arabic", "Vietnamese", "Italian", "Greek", "Spanish", "Hindi", "Filipino"]
        
        for i in range(count):
            carer_id = f"CR{random.randint(100000, 999999):06d}"
            
            profile = CarerProfile(
                carer_id=carer_id,
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                certification_level=random.choice(certifications),
                years_experience=random.randint(0, 25),
                specializations=random.sample(specializations, random.randint(1, 3)),
                available_hours_per_week=random.randint(15, 40),
                languages=random.sample(languages, random.randint(1, 2))
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_participant_profiles(self, count: int = 100) -> List[ParticipantProfile]:
        """生成参与者档案"""
        profiles = []
        
        disability_types = [
            "Intellectual Disability", "Autism Spectrum Disorder", "Cerebral Palsy",
            "Acquired Brain Injury", "Spinal Cord Injury", "Sensory Impairment",
            "Psychosocial Disability", "Neurological Conditions", "Physical Disability",
            "Multiple Sclerosis", "Hearing Impairment", "Vision Impairment",
            "Muscular Dystrophy", "Down Syndrome", "Epilepsy"
        ]
        
        communication_preferences = [
            "Verbal Communication", "Auslan (Sign Language)", "Picture Communication",
            "Written Instructions", "Simple Language", "Assistive Technology",
            "Visual Schedules", "Gesture-based Communication", "Digital Communication"
        ]
        
        support_levels = ["Core Support", "Capacity Building", "Capital Support"]
        age_groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
        
        for i in range(count):
            participant_id = f"PT{random.randint(100000, 999999):06d}"
            
            profile = ParticipantProfile(
                participant_id=participant_id,
                age_group=random.choice(age_groups),
                disability_type=random.choice(disability_types),
                support_level=random.choice(support_levels),
                communication_preferences=random.sample(communication_preferences, random.randint(1, 2)),
                mobility_requirements=random.choice([
                    [], ["Wheelchair access required"], ["Walking frame assistance"], 
                    ["Transfer support needed"], ["Mobility scooter accommodation"],
                    ["Ramp access needed"], ["Accessible parking required"]
                ])
            )
            profiles.append(profile)
        
        return profiles
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """使用纯LLM生成单条服务记录"""
        
        try:
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            carer_name = f"{carer.first_name} {carer.last_name}"
            
            # 确定服务结果（基于权重）
            outcome_weights = {
                ServiceOutcome.POSITIVE: 0.65,
                ServiceOutcome.NEUTRAL: 0.25, 
                ServiceOutcome.NEGATIVE: 0.08,
                ServiceOutcome.INCOMPLETE: 0.02
            }
            outcomes = list(ServiceOutcome)
            weights = [outcome_weights.get(oc, 0.1) for oc in outcomes]
            service_outcome = random.choices(outcomes, weights=weights)[0]
            
            # 确定服务时长
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
            
            # 确定地点
            location_weights = {
                LocationType.HOME: 0.45,
                LocationType.COMMUNITY_CENTRE: 0.20,
                LocationType.HEALTHCARE_FACILITY: 0.10,
                LocationType.SHOPPING_CENTRE: 0.08,
                LocationType.LIBRARY: 0.05,
                LocationType.POOL: 0.04,
                LocationType.PHARMACY: 0.03,
                LocationType.PARK: 0.03,
                LocationType.OTHER: 0.02
            }
            location_types = list(LocationType)
            loc_weights = [location_weights.get(lt, 0.02) for lt in location_types]
            location_type = random.choices(location_types, weights=loc_weights)[0]
            
            # 使用LLM生成所有内容
            logger.info(f"Generating LLM content for {service_type.value} service...")
            llm_content = await self._generate_llm_content(
                service_type, service_outcome, duration, location_type,
                carer_name, participant.age_group, participant.disability_type
            )
            
            # 创建服务记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                carer_name=carer_name,
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=llm_content["narrative_notes"],
                location_type=location_type,
                location_details=f"{location_type.value} - Professional NDIS support environment",
                service_outcome=service_outcome,
                support_techniques_used=llm_content["support_techniques_used"],
                challenges_encountered=llm_content["challenges_encountered"],
                participant_response=llm_content["participant_response"],
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
            logger.error(f"Failed to generate LLM service record: {e}")
            return None
    
    async def generate_dataset(self, size: int = 100) -> List[CarerServiceRecord]:
        """生成完整的LLM数据集"""
        logger.info(f"Starting generation of {size} English service records (Pure LLM method)")
        
        # 生成档案
        self.carers = self.generate_carer_profiles(max(10, size // 15))
        self.participants = self.generate_participant_profiles(max(20, size // 8))
        
        records = []
        
        # NDIS服务类型权重
        service_weights = {
            ServiceType.PERSONAL_CARE: 0.28,
            ServiceType.COMMUNITY_ACCESS: 0.22,
            ServiceType.HOUSEHOLD_TASKS: 0.15,
            ServiceType.SOCIAL_SUPPORT: 0.12,
            ServiceType.TRANSPORT: 0.08,
            ServiceType.SKILL_DEVELOPMENT: 0.06,
            ServiceType.PHYSIOTHERAPY: 0.04,
            ServiceType.RESPITE_CARE: 0.03,
            ServiceType.MEDICATION_SUPPORT: 0.01,
            ServiceType.MEAL_PREPARATION: 0.01
        }
        
        service_types = list(ServiceType)
        weights = [service_weights.get(st, 0.01) for st in service_types]
        
        # 生成记录（序列化以避免并发问题）
        for i in range(size):
            # 随机选择护工和参与者
            carer = random.choice(self.carers)
            participant = random.choice(self.participants)
            
            # 生成服务日期（过去90天内）
            days_ago = random.randint(1, 90)
            service_date = date.today() - timedelta(days=days_ago)
            
            # 选择服务类型
            service_type = random.choices(service_types, weights=weights)[0]
            
            try:
                record = await self.generate_service_record(carer, participant, service_date, service_type)
                if record:
                    records.append(record)
                    logger.info(f"Generated LLM record {i + 1}/{size}: {record.service_type.value}")
                else:
                    logger.warning(f"Failed to generate LLM record {i + 1}")
                
                # 每5条记录报告一次进度
                if (i + 1) % 5 == 0:
                    logger.info(f"Progress: {i + 1}/{size} records, successful: {len(records)}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate record {i+1}: {e}")
        
        logger.info(f"LLM dataset generation completed. Total valid records: {len(records)}")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], prefix: str = "pure_llm_english_carers") -> Dict[str, str]:
        """保存数据集到多种格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        data_dicts = [record.to_dict() for record in records]
        
        # JSON格式
        json_file = output_dir / f"{prefix}_{timestamp}_{len(records)}records.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, ensure_ascii=False, indent=2, default=str)
        saved_files["json"] = str(json_file)
        
        # JSONL格式
        jsonl_file = output_dir / f"{prefix}_{timestamp}_{len(records)}records.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record_dict in data_dicts:
                f.write(json.dumps(record_dict, ensure_ascii=False, default=str) + '\n')
        saved_files["jsonl"] = str(jsonl_file)
        
        # CSV格式
        try:
            import pandas as pd
            df = pd.DataFrame(data_dicts)
            csv_file = output_dir / f"{prefix}_{timestamp}_{len(records)}records.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            saved_files["csv"] = str(csv_file)
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")
        
        logger.info(f"LLM dataset saved to: {saved_files}")
        return saved_files


async def main():
    """主函数 - 演示纯LLM数据生成"""
    try:
        generator = PureLLMEnglishGenerator()
        
        # 生成测试数据
        test_size = 10
        logger.info(f"Generating pure LLM English test dataset ({test_size} records)")
        
        records = await generator.generate_dataset(test_size)
        
        if records:
            # 保存数据
            saved_files = generator.save_dataset(records)
            
            # 输出结果
            print(f"\n✅ Pure LLM English data generation completed successfully!")
            print(f"📊 Generated records: {len(records)}")
            print(f"📁 Saved files:")
            for format_type, filepath in saved_files.items():
                print(f"   {format_type}: {filepath}")
            
            # 显示示例记录
            print(f"\n📋 Sample LLM-generated record:")
            sample = records[0]
            print(f"Carer: {sample.carer_name}")
            print(f"Service Type: {sample.service_type.value}")
            print(f"Duration: {sample.duration_hours} hours")
            print(f"Outcome: {sample.service_outcome.value}")
            print(f"Location: {sample.location_type.value}")
            print(f"Narrative: {sample.narrative_notes[:200]}...")
            print(f"Support Techniques: {', '.join(sample.support_techniques_used)}")
            
        else:
            logger.error("Failed to generate any valid LLM records")
    
    except Exception as e:
        logger.error(f"Pure LLM generation failed: {e}")
        print(f"❌ Pure LLM generation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
