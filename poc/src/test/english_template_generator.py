"""
英文模板数据生成器
完全本地化，不依赖任何外部API
专门生成英文NDIS护工数据
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化英文Faker
fake = Faker(['en_AU'])


class EnglishTemplateGenerator:
    """英文模板数据生成器"""
    
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
        
        # 英文叙述模板
        self.narrative_templates = self._initialize_english_templates()
        
        logger.info("English template data generator initialized")
    
    def _initialize_english_templates(self) -> Dict[str, List[str]]:
        """初始化英文叙述模板"""
        return {
            "positive": [
                "Successfully provided {service_type} support to the participant today. The participant demonstrated excellent cooperation and actively engaged throughout the {duration} hour session. Using {technique} approaches, we achieved all planned objectives with outstanding results. The participant showed enthusiasm and made significant progress in their support goals.",
                
                "Delivered comprehensive {service_type} services for the participant in a {location} setting. The participant responded positively to all interventions and showed remarkable improvement in their abilities. Through the application of {technique} methodologies, we facilitated meaningful progress toward their NDIS goals.",
                
                "Provided professional {service_type} assistance during today's {duration} hour session. The participant exhibited high levels of engagement and cooperation throughout all activities. The {technique} approach proved highly effective, resulting in excellent outcomes and exceeded expectations for this support period.",
                
                "Facilitated {service_type} support with exceptional results today. The participant demonstrated outstanding motivation and actively participated in all planned activities. Using evidence-based {technique} strategies, we successfully accomplished all established care objectives with measurable positive outcomes.",
                
                "Conducted {service_type} intervention with remarkable success during the {duration} hour session. The participant showed excellent responsiveness to the {technique} approach and achieved significant milestones. The session exceeded all expectations and contributed substantially to their ongoing development goals."
            ],
            
            "neutral": [
                "Provided routine {service_type} support to the participant today. The session proceeded according to the established care plan with standard levels of engagement. Using {technique} methodologies, we maintained consistency with previous sessions and achieved the planned objectives within the {duration} hour timeframe.",
                
                "Delivered {service_type} services as scheduled during today's session. The participant demonstrated stable cooperation and completed all planned activities. The {technique} approach supported continued progress according to their NDIS support plan with no significant variations from baseline.",
                
                "Conducted {service_type} support session with typical participant engagement. All scheduled activities were completed as planned using {technique} strategies. The participant maintained their usual level of participation throughout the {duration} hour session with consistent outcomes.",
                
                "Implemented {service_type} interventions according to the established support protocols. The participant showed standard levels of cooperation and engagement throughout the session. Using {technique} approaches, we maintained steady progress toward their support goals.",
                
                "Facilitated {service_type} support during today's scheduled session. The participant exhibited consistent behavior patterns and participated appropriately in all activities. The {technique} methodology supported ongoing stability in their care routine."
            ],
            
            "negative": [
                "Attempted to provide {service_type} support under challenging circumstances today. The participant experienced difficulties with engagement and required additional patience and modified approaches. Despite implementing {technique} strategies, progress was limited and requires reassessment of the current support plan.",
                
                "Worked with the participant on {service_type} activities during a difficult session. The participant showed signs of distress and had trouble participating in planned activities. Although {technique} interventions were employed, outcomes were below expectations and follow-up planning is needed.",
                
                "Provided {service_type} support despite significant behavioral challenges during today's session. The participant required extra support and encouragement throughout the {duration} hour period. While {technique} approaches were utilized, limited progress was achieved and care plan review is recommended.",
                
                "Delivered {service_type} services under challenging conditions with the participant showing resistance to activities. Multiple {technique} strategies were attempted with variable success. The session highlighted areas requiring additional support and intervention modifications.",
                
                "Supported the participant with {service_type} activities while managing behavioral escalation. Despite consistent application of {technique} methodologies, engagement remained limited throughout the session. Additional resources and strategy adjustments are required for future sessions."
            ],
            
            "incomplete": [
                "Began {service_type} support session but was unable to complete all planned activities due to participant needs. The session was shortened after {partial_duration} when the participant requested a break. The {technique} approach was partially implemented before session conclusion.",
                
                "Started {service_type} interventions but the session was interrupted due to external circumstances. Approximately half of the planned activities were completed using {technique} strategies before early conclusion. Rescheduling will be arranged to complete remaining objectives.",
                
                "Initiated {service_type} support but the participant became unwell and required session termination. Limited progress was made in the available time using {technique} approaches. Follow-up planning will address the incomplete objectives and participant welfare.",
                
                "Commenced {service_type} activities but environmental factors required early session completion. Partial implementation of {technique} strategies occurred before conclusion. The remaining support goals will be addressed in the next scheduled session.",
                
                "Began {service_type} support session but participant fatigue necessitated early completion. Initial {technique} interventions were implemented successfully before session conclusion. Care plan adjustments may be needed to accommodate participant energy levels."
            ]
        }
    
    def _get_support_techniques(self) -> List[str]:
        """获取支持技术列表"""
        return [
            "person-centered communication", "behavioral reinforcement", "environmental modification",
            "visual prompting", "task breakdown", "positive encouragement", "structured guidance",
            "adaptive equipment utilization", "sensory regulation", "routine establishment",
            "social modeling", "crisis de-escalation", "cognitive behavioral", "trauma-informed care",
            "strength-based approach", "motivational interviewing", "active listening",
            "collaborative planning", "skill development", "independence promotion"
        ]
    
    def _get_location_descriptions(self) -> Dict[LocationType, List[str]]:
        """获取位置描述"""
        return {
            LocationType.HOME: [
                "comfortable home environment", "familiar domestic setting", "private residence",
                "participant's own home", "supportive home environment"
            ],
            LocationType.COMMUNITY_CENTRE: [
                "accessible community center", "local community facility", "community hub",
                "neighborhood center", "public community space"
            ],
            LocationType.HEALTHCARE_FACILITY: [
                "professional healthcare setting", "medical facility", "clinical environment",
                "therapeutic facility", "specialized healthcare center"
            ],
            LocationType.SHOPPING_CENTRE: [
                "local shopping center", "retail environment", "commercial district",
                "shopping complex", "community shopping area"
            ],
            LocationType.LIBRARY: [
                "quiet library setting", "public library", "educational facility",
                "resource center", "calm library environment"
            ],
            LocationType.POOL: [
                "aquatic center", "swimming facility", "therapeutic pool environment",
                "community pool", "accessible swimming venue"
            ],
            LocationType.PHARMACY: [
                "local pharmacy", "community chemist", "medication center",
                "healthcare pharmacy", "accessible pharmacy location"
            ],
            LocationType.PARK: [
                "outdoor park setting", "natural environment", "community parkland",
                "green space", "recreational park area"
            ],
            LocationType.OTHER: [
                "specialized venue", "appropriate facility", "suitable location",
                "designated space", "professional environment"
            ]
        }
    
    def generate_english_narrative(self, 
                                 service_type: ServiceType, 
                                 outcome: ServiceOutcome,
                                 duration: float,
                                 location_type: LocationType) -> str:
        """生成英文护理叙述"""
        
        # 选择模板类型
        outcome_key = outcome.value if outcome.value != "incomplete" else "incomplete"
        templates = self.narrative_templates.get(outcome_key, self.narrative_templates["neutral"])
        
        # 选择随机模板
        template = random.choice(templates)
        
        # 准备替换变量
        technique = random.choice(self._get_support_techniques())
        location_options = self._get_location_descriptions().get(location_type, ["professional setting"])
        location = random.choice(location_options)
        
        # 为incomplete结果生成部分时长
        partial_duration = round(duration * random.uniform(0.3, 0.7), 1) if outcome == ServiceOutcome.INCOMPLETE else duration
        
        # 替换模板变量
        try:
            narrative = template.format(
                service_type=service_type.value.lower(),
                technique=technique,
                duration=duration,
                location=location,
                partial_duration=partial_duration
            )
        except KeyError as e:
            # 如果模板中有未定义的变量，使用简化版本
            logger.warning(f"Template formatting error: {e}")
            narrative = f"Provided {service_type.value.lower()} support to the participant using {technique} strategies. The session was conducted in a {location} with appropriate professional standards maintained throughout."
        
        # 确保叙述长度合适
        if len(narrative) < 100:
            narrative += f" All activities were conducted according to NDIS quality standards with focus on participant dignity and autonomy."
        elif len(narrative) > 500:
            narrative = narrative[:497] + "..."
        
        return narrative
    
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
            "Certificate IV in Mental Health",
            "Diploma of Disability Support",
            "Certificate IV in Allied Health Assistance"
        ]
        
        # 专业化领域
        specializations = [
            "Personal Care Assistance", "Behavioral Support", "Cognitive Rehabilitation", 
            "Physical Disability Support", "Mental Health Recovery", "Aged Care Services",
            "Developmental Disability Support", "Community Participation", "Autism Support",
            "Acquired Brain Injury Support", "Sensory Impairment Support", "Respite Care",
            "Social Skills Development", "Independent Living Skills"
        ]
        
        # 语言能力
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
        """生成英文参与者档案"""
        profiles = []
        
        # NDIS残疾类型
        disability_types = [
            "Intellectual Disability", "Autism Spectrum Disorder", "Cerebral Palsy",
            "Acquired Brain Injury", "Spinal Cord Injury", "Sensory Impairment",
            "Psychosocial Disability", "Neurological Conditions", "Physical Disability",
            "Multiple Sclerosis", "Hearing Impairment", "Vision Impairment",
            "Muscular Dystrophy", "Down Syndrome", "Epilepsy"
        ]
        
        # 沟通偏好
        communication_preferences = [
            "Verbal Communication", "Auslan (Sign Language)", "Picture Communication",
            "Written Instructions", "Simple Language", "Assistive Technology",
            "Visual Schedules", "Gesture-based Communication", "Digital Communication"
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
        """生成单条英文服务记录"""
        
        try:
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
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
            
            # 生成英文叙述
            narrative = self.generate_english_narrative(service_type, service_outcome, duration, location_type)
            
            # 生成支持技术
            support_techniques = random.sample(self._get_support_techniques(), random.randint(2, 4))
            
            # 生成挑战（如果结果不理想）
            challenges = []
            if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
                challenge_options = [
                    "Communication barriers", "Behavioral escalation", "Environmental sensitivities",
                    "Attention and focus difficulties", "Physical limitations", "Equipment malfunctions",
                    "Time constraints", "Resource limitations", "External distractions"
                ]
                challenges = random.sample(challenge_options, random.randint(1, 3))
            
            # 参与者反应
            response_mapping = {
                ServiceOutcome.POSITIVE: ["Highly engaged", "Very cooperative", "Enthusiastic participation", "Exceeded expectations"],
                ServiceOutcome.NEUTRAL: ["Cooperative", "Stable engagement", "Standard participation", "Adequate response"],
                ServiceOutcome.NEGATIVE: ["Required encouragement", "Challenging session", "Limited cooperation", "Additional support needed"],
                ServiceOutcome.INCOMPLETE: ["Session interrupted", "Early completion", "Requires follow-up", "Partial participation"]
            }
            participant_response = random.choice(response_mapping.get(service_outcome, ["Standard response"]))
            
            # 生成位置详情
            location_descriptions = self._get_location_descriptions()
            location_detail = random.choice(location_descriptions.get(location_type, ["Professional environment"]))
            
            # 创建服务记录
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                carer_name=f"{carer.first_name} {carer.last_name}",
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=narrative,
                location_type=location_type,
                location_details=f"{location_detail} - {location_type.value}",
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
        logger.info(f"Starting generation of {size} English service records (template method)")
        
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
        
        # 生成记录
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
                
                if (i + 1) % 25 == 0:
                    logger.info(f"Generated {i + 1} records, successful: {len(records)}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate record {i+1}: {e}")
        
        logger.info(f"Dataset generation completed. Total valid records: {len(records)}")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], prefix: str = "english_template_carers") -> Dict[str, str]:
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
    """主函数 - 演示英文模板数据生成"""
    generator = EnglishTemplateGenerator()
    
    # 生成测试数据
    test_size = 100
    logger.info(f"Generating English test dataset ({test_size} records)")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 输出结果
        print(f"\n✅ English template data generation completed successfully!")
        print(f"📊 Generated records: {len(records)}")
        print(f"📁 Saved files:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
        
        # 显示示例记录
        print(f"\n📋 Sample record:")
        sample = records[0]
        print(f"Service Type: {sample.service_type.value}")
        print(f"Duration: {sample.duration_hours} hours")
        print(f"Outcome: {sample.service_outcome.value}")
        print(f"Location: {sample.location_type.value}")
        print(f"Narrative: {sample.narrative_notes[:300]}...")
        
    else:
        logger.error("Failed to generate any valid records")


if __name__ == "__main__":
    asyncio.run(main())
