"""
完全本地的英文数据生成器
不依赖任何外部API，生成高质量英文护工数据
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from faker import Faker

from english_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, EnglishDataValidator
)
from config import get_config
from data_validator import ComprehensiveValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker（英文/澳大利亚本地化）
fake = Faker(['en_AU'])


class LocalEnglishGenerator:
    """完全本地的英文数据生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 初始化数据验证器
        self.validator = EnglishDataValidator()
        
        # 预生成档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        # 预定义的英文叙述组件
        self.narrative_components = self._initialize_narrative_components()
        
        logger.info("本地英文数据生成器初始化完成")
    
    def _initialize_narrative_components(self) -> Dict[str, Dict[str, List[str]]]:
        """初始化英文叙述组件"""
        return {
            "openings": {
                "positive": [
                    "Successfully provided {service_type} support to {participant_name}.",
                    "Delivered excellent {service_type} services for {participant_name} today.",
                    "Facilitated comprehensive {service_type} assistance to {participant_name}.",
                    "Provided professional {service_type} support for {participant_name}.",
                    "Effectively delivered {service_type} services to {participant_name}."
                ],
                "neutral": [
                    "Provided routine {service_type} support to {participant_name}.",
                    "Delivered standard {service_type} services for {participant_name}.",
                    "Conducted {service_type} session with {participant_name}.",
                    "Assisted {participant_name} with {service_type} activities.",
                    "Carried out {service_type} support for {participant_name}."
                ],
                "negative": [
                    "Attempted to provide {service_type} support to {participant_name} with challenges.",
                    "Worked with {participant_name} on {service_type} under difficult circumstances.",
                    "Provided {service_type} support to {participant_name} despite obstacles.",
                    "Delivered {service_type} services to {participant_name} with limited success.",
                    "Supported {participant_name} with {service_type} activities facing difficulties."
                ]
            },
            "participant_responses": {
                "positive": [
                    "The participant demonstrated excellent cooperation and actively engaged throughout the session.",
                    "Participant showed strong motivation and willingness to participate in all activities.",
                    "The participant responded positively to guidance and demonstrated improved skills.",
                    "Participant actively participated and showed enthusiasm for the planned activities.",
                    "The participant exhibited outstanding cooperation and achieved all set objectives."
                ],
                "neutral": [
                    "The participant maintained stable engagement and completed activities as planned.",
                    "Participant demonstrated consistent cooperation and followed established routines.",
                    "The participant showed average engagement and completed most planned tasks.",
                    "Participant remained calm and cooperative throughout the session.",
                    "The participant exhibited standard levels of participation and compliance."
                ],
                "negative": [
                    "The participant experienced emotional dysregulation and showed resistance to activities.",
                    "Participant demonstrated difficulty with attention and cooperation during the session.",
                    "The participant required extra patience and modified intervention approaches.",
                    "Participant showed signs of distress and limited engagement with planned activities.",
                    "The participant exhibited challenging behaviors requiring specialized support strategies."
                ]
            },
            "techniques": {
                "positive": [
                    "Through effective implementation of {technique} strategies, we successfully accomplished established care goals.",
                    "Using evidence-based {technique} approaches, we achieved meaningful and measurable outcomes.",
                    "The application of {technique} methodologies proved highly effective in meeting therapeutic objectives.",
                    "By utilizing {technique} techniques, we facilitated significant progress toward care plan goals.",
                    "Implementation of {technique} interventions resulted in excellent therapeutic outcomes."
                ],
                "neutral": [
                    "Standard {technique} protocols were applied and the session proceeded as planned.",
                    "Using routine {technique} approaches, we maintained steady progress according to care plans.",
                    "The {technique} methodology supported continued stability and consistency of care.",
                    "Applied {technique} strategies to maintain established routines and expectations.",
                    "Routine {technique} techniques were employed to support ongoing care objectives."
                ],
                "negative": [
                    "Despite implementing {technique} interventions, progress was limited and requires review.",
                    "While {technique} strategies were employed, outcomes were below expectations.",
                    "The {technique} approach was partially effective but additional strategies are needed.",
                    "Although {technique} methods were utilized, challenges persisted requiring alternative approaches.",
                    "Implementation of {technique} techniques had limited success and warrants care plan revision."
                ]
            },
            "locations": [
                "participant's home environment", "community center facility", "designated therapy room",
                "quiet support area", "familiar indoor setting", "accessible community venue",
                "private consultation room", "specialized treatment space", "comfortable meeting area",
                "appropriate clinical environment"
            ],
            "support_techniques": [
                "person-centered communication", "behavioral reinforcement", "environmental modification",
                "visual prompting", "task breakdown", "positive encouragement", "structured guidance",
                "adaptive equipment", "sensory regulation", "routine establishment", "social modeling",
                "crisis de-escalation", "cognitive behavioral", "trauma-informed care", "strength-based approach"
            ],
            "challenges": [
                "communication barriers", "behavioral escalation", "environmental sensitivities",
                "attention and focus difficulties", "emotional regulation challenges", "physical limitations",
                "cognitive processing delays", "social interaction concerns", "medication side effects",
                "family dynamics impact", "equipment malfunctions", "scheduling conflicts"
            ]
        }
    
    def generate_english_narrative(self, 
                                 service_type: ServiceType, 
                                 outcome: ServiceOutcome,
                                 participant_name: str = None) -> str:
        """生成专业英文护理叙述"""
        
        participant_name = participant_name or fake.first_name()
        service_type_text = service_type.value.lower()
        outcome_key = outcome.value if outcome.value != "incomplete" else "negative"
        
        # 选择叙述组件
        opening = random.choice(self.narrative_components["openings"][outcome_key])
        response = random.choice(self.narrative_components["participant_responses"][outcome_key])
        technique_name = random.choice(self.narrative_components["support_techniques"])
        technique_desc = random.choice(self.narrative_components["techniques"][outcome_key])
        location = random.choice(self.narrative_components["locations"])
        
        # 构建完整叙述
        narrative_parts = [
            opening.format(service_type=service_type_text, participant_name=participant_name),
            response,
            technique_desc.format(technique=technique_name),
            f"The session was conducted in the {location} to ensure optimal support delivery."
        ]
        
        # 根据结果添加额外信息
        if outcome == ServiceOutcome.POSITIVE:
            narrative_parts.append("Overall, the session exceeded expectations and contributed to meaningful progress.")
        elif outcome == ServiceOutcome.NEGATIVE or outcome == ServiceOutcome.INCOMPLETE:
            narrative_parts.append("Follow-up planning and strategy adjustment will be prioritized.")
        else:
            narrative_parts.append("The session maintained continuity of care and supported ongoing objectives.")
        
        # 组合并调整长度
        full_narrative = " ".join(narrative_parts)
        
        # 确保长度在合理范围内
        if len(full_narrative) > 800:
            # 保留前三个部分
            full_narrative = " ".join(narrative_parts[:3])
            if len(full_narrative) > 800:
                full_narrative = full_narrative[:797] + "..."
        elif len(full_narrative) < 100:
            full_narrative += " The intervention was completed according to established protocols."
        
        return full_narrative
    
    def generate_carer_profiles(self, count: int = 50) -> List[CarerProfile]:
        """生成英文护工档案"""
        profiles = []
        
        # 澳大利亚护理资格
        certifications = [
            "Certificate III in Individual Support", 
            "Certificate IV in Disability Support",
            "Diploma of Community Services", 
            "Bachelor of Nursing",
            "Certificate III in Aged Care",
            "Certificate IV in Mental Health"
        ]
        
        # 英文专业化领域
        specializations = [
            "Personal Care Assistance", "Behavioral Support", "Cognitive Rehabilitation", 
            "Physical Disability Support", "Mental Health Recovery", "Aged Care Services",
            "Developmental Disability Support", "Community Participation", "Autism Support",
            "Acquired Brain Injury Support", "Sensory Impairment Support"
        ]
        
        # 澳大利亚常用语言
        languages = ["English", "Mandarin", "Arabic", "Vietnamese", "Italian", "Greek", "Spanish", "Hindi"]
        
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
        """生成英文参与者档案"""
        profiles = []
        
        # NDIS标准残疾类型
        disability_types = [
            "Intellectual Disability", "Autism Spectrum Disorder", "Cerebral Palsy",
            "Acquired Brain Injury", "Spinal Cord Injury", "Sensory Impairment",
            "Psychosocial Disability", "Neurological Conditions", "Physical Disability",
            "Multiple Sclerosis", "Hearing Impairment", "Vision Impairment"
        ]
        
        # 沟通偏好
        communication_preferences = [
            "Verbal Communication", "Auslan (Sign Language)", "Picture Communication",
            "Written Instructions", "Simple Language", "Assistive Technology",
            "Visual Schedules", "Gesture-based Communication"
        ]
        
        # 支持级别
        support_levels = ["Core Support", "Capacity Building", "Capital Support"]
        
        # 年龄组
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
                    ["Transfer support needed"], ["Mobility scooter accommodation"]
                ])
            )
            profiles.append(profile)
        
        return profiles
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """生成单条英文服务记录"""
        
        try:
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
            # 确定服务结果（基于权重）
            outcome_weights = {
                "positive": 0.65,
                "neutral": 0.25, 
                "negative": 0.08,
                "incomplete": 0.02
            }
            outcomes = list(ServiceOutcome)
            weights = [outcome_weights.get(oc.value, 0.1) for oc in outcomes]
            service_outcome = random.choices(outcomes, weights=weights)[0]
            
            # 生成英文叙述
            participant_name = fake.first_name()
            narrative = self.generate_english_narrative(service_type, service_outcome, participant_name)
            
            # 确定服务时长（基于澳大利亚NDIS标准）
            duration_ranges = {
                "Personal Care": (0.5, 4.0),
                "Household Tasks": (1.0, 6.0),
                "Community Access": (2.0, 8.0),
                "Transport Assistance": (0.5, 3.0),
                "Social Support": (1.0, 4.0),
                "Physiotherapy": (0.5, 2.0),
                "Medication Support": (0.25, 1.0),
                "Skill Development": (1.0, 6.0),
                "Respite Care": (2.0, 8.0),
                "Meal Preparation": (0.5, 2.0)
            }
            duration_range = duration_ranges.get(service_type.value, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            # 确定地点（权重分布）
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
            
            # 生成支持技术
            support_techniques = random.sample(
                self.narrative_components["support_techniques"], 
                random.randint(2, 4)
            )
            
            # 生成挑战（如果结果不理想）
            challenges = []
            if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
                challenges = random.sample(
                    self.narrative_components["challenges"],
                    random.randint(1, 3)
                )
            
            # 参与者反应
            response_mapping = {
                ServiceOutcome.POSITIVE: ["Highly engaged", "Very cooperative", "Enthusiastic participation", "Exceeded expectations"],
                ServiceOutcome.NEUTRAL: ["Cooperative", "Stable engagement", "Standard participation", "Adequate response"],
                ServiceOutcome.NEGATIVE: ["Required encouragement", "Challenging session", "Limited cooperation", "Additional support needed"],
                ServiceOutcome.INCOMPLETE: ["Session interrupted", "Early completion", "Requires follow-up", "Partial participation"]
            }
            participant_response = random.choice(response_mapping.get(service_outcome, ["Standard response"]))
            
            # 创建服务记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=narrative,
                location_type=location_type,
                location_details=f"{location_type.value} - Professional support environment",
                service_outcome=service_outcome,
                support_techniques_used=support_techniques,
                challenges_encountered=challenges,
                participant_response=participant_response,
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
            logger.error(f"Failed to generate service record: {e}")
            return None
    
    async def generate_dataset(self, size: int = 100) -> List[CarerServiceRecord]:
        """生成完整的英文数据集"""
        logger.info(f"Starting generation of {size} English service records (local method)")
        
        # 生成档案
        self.carers = self.generate_carer_profiles(max(10, size // 15))
        self.participants = self.generate_participant_profiles(max(20, size // 8))
        
        records = []
        
        # NDIS服务类型权重（基于真实使用模式）
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
        
        # 生成记录
        for i in range(size):
            # 随机选择护工和参与者
            carer = random.choice(self.carers)
            participant = random.choice(self.participants)
            
            # 生成服务日期（过去90天内，工作日概率更高）
            days_ago = random.randint(1, 90)
            service_date = date.today() - timedelta(days=days_ago)
            
            # 调整为工作日的概率更高
            if service_date.weekday() >= 5:  # 周末
                if random.random() < 0.3:  # 30%概率调整到工作日
                    days_adjust = random.randint(1, 2)
                    service_date = service_date - timedelta(days=days_adjust)
            
            # 选择服务类型
            service_type = random.choices(service_types, weights=weights)[0]
            
            try:
                record = await self.generate_service_record(carer, participant, service_date, service_type)
                if record:
                    records.append(record)
                
                if (i + 1) % 25 == 0:
                    logger.info(f"Generated {i + 1} records, successful: {len(records)}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate record {i+1}: {e}")
        
        logger.info(f"Dataset generation completed. Total valid records: {len(records)}")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], prefix: str = "local_english_carers") -> Dict[str, str]:
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
        
        logger.info(f"Dataset saved to: {saved_files}")
        return saved_files


async def main():
    """主函数 - 演示本地英文数据生成"""
    generator = LocalEnglishGenerator()
    
    # 生成测试数据
    test_size = 100
    logger.info(f"Generating local English test dataset ({test_size} records)")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 执行验证
        logger.info("Performing data validation...")
        validator = ComprehensiveValidator()
        validation_results = validator.comprehensive_validation(records)
        
        # 保存验证报告
        report_file = validator.save_validation_report(
            validation_results, 
            f"local_english_validation_{test_size}records.json"
        )
        
        # 输出结果
        print(f"\n✅ Local English data generation completed successfully!")
        print(f"📊 Generated records: {len(records)}")
        print(f"🎯 Quality score: {validation_results['overall_score']}/100")
        print(f"🔒 Privacy score: {validation_results['privacy_analysis']['anonymization_score']}/100")
        print(f"📁 Saved files:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
        print(f"📋 Validation report: {report_file}")
        
        # 显示示例记录
        print(f"\n📋 Sample record:")
        print(records[0].to_json())
        
    else:
        logger.error("Failed to generate any valid records")


if __name__ == "__main__":
    asyncio.run(main())

