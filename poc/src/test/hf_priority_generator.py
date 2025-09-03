"""
优先使用Hugging Face的免费数据生成器
当Ollama不可用时，使用Hugging Face API
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

from faker import Faker
from tenacity import retry, stop_after_attempt, wait_random_exponential

from english_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, EnglishDataValidator
)
from config import get_config
from data_validator import ComprehensiveValidator
from free_config import FREE_LLM_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker
fake = Faker(['en_AU'])


class HuggingFacePriorityGenerator:
    """优先使用Hugging Face的免费数据生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 选择可用的生成方法
        self.active_method = self._select_generation_method()
        
        # 初始化数据验证器
        self.validator = EnglishDataValidator()
        
        # 预生成档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info(f"Hugging Face优先生成器初始化完成，使用方法: {self.active_method}")
    
    def _select_generation_method(self) -> str:
        """选择可用的生成方法"""
        
        # 1. 检查Hugging Face
        hf_token = FREE_LLM_CONFIG.get("huggingface", {}).get("token", "")
        if hf_token and hf_token != "your_huggingface_token_here":
            if self._test_huggingface_connection(hf_token):
                return "huggingface"
            else:
                logger.warning("Hugging Face token配置但连接失败")
        
        # 2. 检查Ollama（如果HF不可用）
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    # 测试一个简单的生成请求
                    if self._test_ollama_generation():
                        return "ollama"
                    else:
                        logger.warning("Ollama可用但生成测试失败")
        except:
            logger.info("Ollama不可用")
        
        # 3. 回退到模板生成
        logger.info("使用模板生成方法作为备选")
        return "template"
    
    def _test_huggingface_connection(self, token: str) -> bool:
        """测试Hugging Face连接"""
        try:
            api_url = "https://api-inference.huggingface.co/models/gpt2"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(api_url, headers=headers, timeout=10)
            return response.status_code in [200, 503]  # 503表示模型正在加载
        except:
            return False
    
    def _test_ollama_generation(self) -> bool:
        """测试Ollama生成"""
        try:
            data = {
                "model": "gpt-oss:20b",  # 您系统中的模型
                "prompt": "Test",
                "stream": False,
                "options": {"max_tokens": 10}
            }
            response = requests.post(
                "http://localhost:11434/api/generate", 
                json=data, 
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    async def _call_huggingface(self, prompt: str) -> str:
        """调用Hugging Face API"""
        hf_config = FREE_LLM_CONFIG["huggingface"]
        token = hf_config["token"]
        
        # 使用GPT-2进行生成
        api_url = "https://api-inference.huggingface.co/models/gpt2"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": 180,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False,
                "pad_token_id": 50256
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    return generated_text.strip()
                return ""
            elif response.status_code == 503:
                # 模型正在加载，稍等重试
                await asyncio.sleep(20)
                raise Exception("Model loading, retrying...")
            else:
                raise Exception(f"Hugging Face API错误: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Hugging Face调用失败: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    async def _call_ollama(self, prompt: str) -> str:
        """调用Ollama本地模型"""
        try:
            data = {
                "model": "gpt-oss:20b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 200
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
                raise Exception(f"Ollama API错误: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise
    
    def _generate_with_template(self, service_type: ServiceType, outcome: ServiceOutcome) -> str:
        """使用模板生成英文叙述"""
        
        participant_name = fake.first_name()
        
        # 根据服务类型和结果生成叙述
        if outcome == ServiceOutcome.POSITIVE:
            templates = [
                f"Provided excellent {service_type.value.lower()} support for {participant_name}. The participant demonstrated strong cooperation and actively engaged in all planned activities. Through effective implementation of person-centered strategies, we successfully accomplished the established care goals.",
                
                f"Delivered professional {service_type.value.lower()} services to {participant_name} today. The participant showed motivation and willingness to participate. Using evidence-based approaches, we achieved meaningful outcomes and the participant expressed satisfaction.",
                
                f"Facilitated {service_type.value.lower()} activities for {participant_name} with outstanding results. The participant responded positively to guidance and demonstrated improved independence. The session was highly effective in meeting therapeutic objectives."
            ]
        elif outcome == ServiceOutcome.NEUTRAL:
            templates = [
                f"Provided routine {service_type.value.lower()} support for {participant_name}. The participant maintained stable engagement and completed activities as planned. Standard care protocols were followed and the session proceeded normally.",
                
                f"Delivered {service_type.value.lower()} services to {participant_name} according to the care plan. The participant demonstrated consistent cooperation and followed established routines. Progress was steady and in line with expectations.",
                
                f"Assisted {participant_name} with {service_type.value.lower()} activities in a structured manner. The participant showed average engagement and completed most planned tasks. The session maintained continuity of care."
            ]
        else:  # NEGATIVE or INCOMPLETE
            templates = [
                f"Attempted to provide {service_type.value.lower()} support for {participant_name} but encountered challenges. The participant experienced some difficulties and required additional encouragement. Modified approaches were implemented and follow-up is planned.",
                
                f"Provided {service_type.value.lower()} services to {participant_name} with mixed outcomes. The participant showed some resistance and required extra support. Alternative strategies were employed and the care plan may need review.",
                
                f"Supported {participant_name} with {service_type.value.lower()} activities under challenging circumstances. The participant needed additional time and patience. While progress was limited, important insights were gained for future sessions."
            ]
        
        return random.choice(templates)
    
    async def generate_narrative(self, service_type: ServiceType, outcome: ServiceOutcome) -> str:
        """生成护理叙述"""
        
        if self.active_method == "huggingface":
            # 构建提示
            participant_name = fake.first_name()
            outcome_desc = "successful" if outcome == ServiceOutcome.POSITIVE else "challenging" if outcome == ServiceOutcome.NEGATIVE else "routine"
            
            prompt = f"The carer provided {service_type.value.lower()} support to {participant_name} with a {outcome_desc} outcome. The participant"
            
            try:
                generated = await self._call_huggingface(prompt)
                if generated:
                    full_narrative = f"{prompt} {generated}"
                    # 确保长度合适
                    if len(full_narrative) > 800:
                        full_narrative = full_narrative[:800] + "..."
                    if len(full_narrative) < 50:
                        full_narrative += " The session was completed successfully."
                    return full_narrative
            except Exception as e:
                logger.warning(f"Hugging Face生成失败，回退到模板: {e}")
        
        elif self.active_method == "ollama":
            participant_name = fake.first_name()
            prompt = f"Write a professional carer service record for {service_type.value.lower()} support provided to {participant_name}. The outcome was {outcome.value}. Include specific details about the service delivery:"
            
            try:
                generated = await self._call_ollama(prompt)
                if generated and len(generated) > 50:
                    return generated[:800]  # 限制长度
            except Exception as e:
                logger.warning(f"Ollama生成失败，回退到模板: {e}")
        
        # 回退到模板生成
        return self._generate_with_template(service_type, outcome)
    
    def generate_carer_profiles(self, count: int = 50) -> List[CarerProfile]:
        """生成护工档案"""
        profiles = []
        config = self.config["carer_profile"]
        
        # 英文专业化
        english_specializations = [
            "Personal Care", "Behavioral Support", "Cognitive Support", 
            "Physical Disability Support", "Mental Health Support", 
            "Aged Care", "Developmental Support", "Community Access"
        ]
        
        for i in range(count):
            carer_id = f"CR{random.randint(100000, 999999):06d}"
            
            profile = CarerProfile(
                carer_id=carer_id,
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                certification_level=random.choice(config["certification_levels"]),
                years_experience=random.randint(*config["experience_range"]),
                specializations=random.sample(english_specializations, random.randint(1, 3)),
                available_hours_per_week=random.randint(*config["hours_range"]),
                languages=random.sample(config["languages"], random.randint(1, 2))
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_participant_profiles(self, count: int = 100) -> List[ParticipantProfile]:
        """生成参与者档案"""
        profiles = []
        config = self.config["participant_profile"]
        
        english_disability_types = [
            "Intellectual Disability", "Autism Spectrum Disorder", 
            "Physical Disability", "Sensory Disability", 
            "Psychosocial Disability", "Neurological Disability", 
            "Multiple Disabilities", "Acquired Brain Injury"
        ]
        
        english_communication = [
            "Verbal Communication", "Sign Language", "Picture Exchange", 
            "Written Communication", "Assistive Technology", "Simple Language"
        ]
        
        for i in range(count):
            participant_id = f"PT{random.randint(100000, 999999):06d}"
            
            profile = ParticipantProfile(
                participant_id=participant_id,
                age_group=random.choice(config["age_groups"]),
                disability_type=random.choice(english_disability_types),
                support_level=random.choice(config["support_levels"]),
                communication_preferences=random.sample(english_communication, random.randint(1, 2)),
                mobility_requirements=random.choice([
                    [], ["wheelchair"], ["walking aid"], ["transfer assistance"]
                ])
            )
            profiles.append(profile)
        
        return profiles
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """生成单条服务记录"""
        
        try:
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
            # 确定服务结果
            outcome_weights = self.config["service"]["outcome_weights"]
            outcomes = list(ServiceOutcome)
            weights = [outcome_weights.get(oc.value, 0.1) for oc in outcomes]
            service_outcome = random.choices(outcomes, weights=weights)[0]
            
            # 生成叙述
            narrative = await self.generate_narrative(service_type, service_outcome)
            
            # 其他字段
            duration_ranges = self.config["service"]["duration_ranges"]
            duration_range = duration_ranges.get(service_type.value, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            location_weights = self.config["location"]["location_weights"]
            location_types = list(LocationType)
            loc_weights = [location_weights.get(lt.value, 0.01) for lt in location_types]
            location_type = random.choices(location_types, weights=loc_weights)[0]
            
            # 支持技术
            support_techniques = random.sample([
                "Visual Prompts", "Verbal Guidance", "Physical Assistance", "Environmental Modification",
                "Behavioral Reinforcement", "Sensory Support", "Time Management", "Social Skills Training"
            ], random.randint(2, 4))
            
            # 创建记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=narrative,
                location_type=location_type,
                location_details=f"{location_type.value} - Designated support area",
                service_outcome=service_outcome,
                support_techniques_used=support_techniques,
                challenges_encountered=[],
                participant_response="Cooperative" if service_outcome == ServiceOutcome.POSITIVE else "Stable",
                follow_up_required=service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]
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
    
    async def generate_dataset(self, size: int = 100) -> List[CarerServiceRecord]:
        """生成完整数据集"""
        logger.info(f"开始生成 {size} 条英文服务记录（使用{self.active_method}）")
        
        # 生成档案
        self.carers = self.generate_carer_profiles(max(10, size // 20))
        self.participants = self.generate_participant_profiles(max(20, size // 10))
        
        records = []
        
        # 服务类型权重
        service_weights = self.config["service"]["service_types_weights"]
        service_types = list(ServiceType)
        weights = [service_weights.get(st.value, 0.1) for st in service_types]
        
        # 生成记录
        for i in range(size):
            carer = random.choice(self.carers)
            participant = random.choice(self.participants)
            
            days_ago = random.randint(1, 90)
            service_date = date.today() - timedelta(days=days_ago)
            
            service_type = random.choices(service_types, weights=weights)[0]
            
            try:
                record = await self.generate_service_record(carer, participant, service_date, service_type)
                if record:
                    records.append(record)
                
                if (i + 1) % 25 == 0:
                    logger.info(f"已生成 {i + 1} 条记录，成功 {len(records)} 条")
                    
            except Exception as e:
                logger.warning(f"生成第 {i+1} 条记录失败: {e}")
        
        logger.info(f"数据集生成完成，共 {len(records)} 条有效记录")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], prefix: str = "hf_carers_data") -> Dict[str, str]:
        """保存数据集"""
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
            logger.warning("pandas未安装，跳过CSV导出")
        
        logger.info(f"数据集已保存到: {saved_files}")
        return saved_files


async def main():
    """主函数"""
    generator = HuggingFacePriorityGenerator()
    
    # 生成测试数据
    test_size = 20
    logger.info(f"生成测试数据集（{test_size}条记录）")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 验证
        validator = ComprehensiveValidator()
        validation_results = validator.comprehensive_validation(records)
        
        # 输出结果
        print(f"\n✅ Hugging Face数据生成完成!")
        print(f"📊 生成记录数: {len(records)}")
        print(f"🔧 使用方法: {generator.active_method}")
        print(f"🎯 质量评分: {validation_results['overall_score']}/100")
        print(f"📁 保存的文件:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
        
        # 显示示例记录
        if records:
            print(f"\n📋 示例记录:")
            print(records[0].to_json())
        
    else:
        logger.error("未能生成任何有效记录")


if __name__ == "__main__":
    asyncio.run(main())

