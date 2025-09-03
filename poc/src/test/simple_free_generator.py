"""
简化的免费数据生成器
专门处理中文护理叙述，避免模板解析问题
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from faker import Faker

from carer_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, DataValidator
)
from config import get_config
from data_validator import ComprehensiveValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker
fake = Faker(['en_AU', 'zh_CN'])


class SimpleFreeGenerator:
    """简化的免费数据生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 初始化数据验证器
        self.validator = DataValidator()
        
        # 预生成的护工和参与者档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info("简化免费数据生成器初始化完成")
    
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
        
        return profiles
    
    def generate_narrative(self, 
                          service_type: ServiceType, 
                          outcome: ServiceOutcome,
                          participant_name: str = None) -> str:
        """生成护理叙述"""
        
        participant_name = participant_name or fake.first_name()
        
        # 护理技术和方法
        techniques = [
            "渐进式引导", "正向强化", "结构化支持", "感官调节",
            "认知重构", "行为塑造", "环境适应", "沟通辅助",
            "个体化支持", "团队协作", "多感官刺激", "行为干预"
        ]
        
        # 服务地点
        locations = [
            "参与者家中", "社区中心活动室", "康复训练室", 
            "户外花园", "安静的房间", "专用治疗区域",
            "熟悉的环境", "日间护理中心"
        ]
        
        # 根据服务类型和结果生成叙述
        service_type_cn = service_type.value
        technique = random.choice(techniques)
        location = random.choice(locations)
        
        if outcome == ServiceOutcome.POSITIVE:
            narratives = [
                f"护工为参与者{participant_name}提供{service_type_cn}服务。参与者积极配合，表现出良好的参与度。护工采用{technique}方法，在{location}进行专业支持。整个服务过程顺利，达到了预期的护理目标，参与者满意度高。",
                f"今日为参与者{participant_name}实施{service_type_cn}支持计划。参与者主动参与各项活动，配合度极佳。护工运用{technique}策略，确保服务质量。在{location}的环境下，成功完成了所有既定目标。",
                f"护工协助参与者{participant_name}进行{service_type_cn}活动。参与者情绪稳定，积极响应护工的指导。通过{technique}技术的有效应用，在{location}取得了显著的护理效果。",
                f"为参与者{participant_name}提供专业的{service_type_cn}服务。参与者表现出色，能够很好地理解并执行护工的指导。采用{technique}方法，在{location}创造了良好的护理环境，达到预期效果。"
            ]
        elif outcome == ServiceOutcome.NEUTRAL:
            narratives = [
                f"护工为参与者{participant_name}提供{service_type_cn}服务。参与者表现平稳，按计划完成了基本的护理活动。护工使用{technique}方法，在{location}进行常规支持。整体进展正常，无特殊情况。",
                f"今日协助参与者{participant_name}进行{service_type_cn}。参与者状态稳定，配合度一般。护工采用{technique}策略提供支持，在{location}按既定流程执行护理计划。",
                f"护工为参与者{participant_name}实施{service_type_cn}支持。参与者反应平常，能够配合完成必要的活动。通过{technique}技术，在{location}维持了稳定的护理标准。",
                f"为参与者{participant_name}提供{service_type_cn}服务支持。参与者表现平静，护工采用{technique}方法进行干预。在{location}的环境下，活动按计划正常进行。"
            ]
        else:  # NEGATIVE or INCOMPLETE
            narratives = [
                f"护工为参与者{participant_name}提供{service_type_cn}服务时遇到挑战。参与者情绪波动较大，对某些活动表现出抗拒。护工耐心采用{technique}方法进行安抚，在{location}尽力创造支持性环境。需要调整策略并安排后续跟进。",
                f"今日为参与者{participant_name}实施{service_type_cn}支持遇到困难。参与者注意力不集中，配合度有限。护工运用{technique}策略尝试引导，但效果不够理想。需要重新评估护理计划。",
                f"护工协助参与者{participant_name}进行{service_type_cn}活动时面临挑战。参与者需要额外的耐心和支持。虽然采用了{technique}技术，但在{location}的服务效果有限，需要多专业团队协作。",
                f"为参与者{participant_name}提供{service_type_cn}服务过程中出现困难。参与者状态不稳定，护工使用{technique}方法进行干预。需要更多时间和个性化支持策略。"
            ]
        
        return random.choice(narratives)
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """生成单条服务记录"""
        
        try:
            # 生成基础记录数据
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
            # 确定服务结果（基于权重）
            outcome_weights = self.config["service"]["outcome_weights"]
            outcomes = list(ServiceOutcome)
            weights = [outcome_weights.get(oc.value, 0.1) for oc in outcomes]
            service_outcome = random.choices(outcomes, weights=weights)[0]
            
            # 确定服务时长
            duration_ranges = self.config["service"]["duration_ranges"]
            duration_range = duration_ranges.get(service_type.value, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            # 确定地点
            location_weights = self.config["location"]["location_weights"]
            location_types = list(LocationType)
            loc_weights = [location_weights.get(lt.value, 0.01) for lt in location_types]
            location_type = random.choices(location_types, weights=loc_weights)[0]
            
            # 生成叙述
            participant_name = fake.first_name()
            narrative = self.generate_narrative(service_type, service_outcome, participant_name)
            
            # 生成支持技术和挑战
            support_techniques = random.sample([
                "视觉提示", "口语指导", "物理协助", "环境调整",
                "行为强化", "感官支持", "时间管理", "社交技能训练",
                "个性化沟通", "情绪调节", "认知训练", "功能性活动"
            ], random.randint(2, 4))
            
            challenges = []
            if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
                challenges = random.sample([
                    "参与者情绪管理", "环境适应困难", "沟通障碍",
                    "注意力维持", "体力限制", "认知负荷",
                    "感官敏感", "行为表现", "社交互动"
                ], random.randint(1, 3))
            
            # 参与者反应
            participant_responses = {
                ServiceOutcome.POSITIVE: ["积极配合", "主动参与", "表现出色", "满意度高"],
                ServiceOutcome.NEUTRAL: ["基本配合", "表现稳定", "正常参与", "状态平稳"],
                ServiceOutcome.NEGATIVE: ["需要鼓励", "情绪波动", "需要支持", "表现困难"],
                ServiceOutcome.INCOMPLETE: ["需要休息", "注意力分散", "状态不佳", "需要调整"]
            }
            
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
                location_details=f"{location_type.value}的专门区域",
                service_outcome=service_outcome,
                support_techniques_used=support_techniques,
                challenges_encountered=challenges,
                participant_response=random.choice(participant_responses.get(service_outcome, ["正常反应"])),
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
    
    async def generate_dataset(self, size: int = 1000) -> List[CarerServiceRecord]:
        """生成完整数据集"""
        logger.info(f"开始生成 {size} 条服务记录")
        
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
                
                if (i + 1) % 50 == 0:
                    logger.info(f"已生成 {i + 1} 条记录，成功 {len(records)} 条")
                    
            except Exception as e:
                logger.warning(f"生成第 {i+1} 条记录失败: {e}")
        
        logger.info(f"数据集生成完成，共 {len(records)} 条有效记录")
        return records
    
    def save_dataset(self, 
                    records: List[CarerServiceRecord], 
                    filename_prefix: str = "simple_free_carers_data") -> Dict[str, str]:
        """保存数据集"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
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
    """主函数 - 演示简化免费数据生成"""
    generator = SimpleFreeGenerator()
    
    # 生成测试数据
    test_size = 100
    logger.info(f"生成测试数据集（{test_size}条记录）")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 进行验证
        logger.info("执行数据验证...")
        validator = ComprehensiveValidator()
        validation_results = validator.comprehensive_validation(records)
        
        # 保存验证报告
        report_file = validator.save_validation_report(
            validation_results, 
            f"simple_free_validation_report_{test_size}records.json"
        )
        
        # 输出结果
        print(f"\n✅ 简化免费数据生成完成!")
        print(f"📊 生成记录数: {len(records)}")
        print(f"🎯 质量评分: {validation_results['overall_score']}/100")
        print(f"🔒 隐私评分: {validation_results['privacy_analysis']['anonymization_score']}/100")
        print(f"📁 保存的文件:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
        print(f"📋 验证报告: {report_file}")
        
        # 显示示例记录
        print(f"\n📋 示例记录:")
        print(records[0].to_json())
        
    else:
        logger.error("未能生成任何有效记录")


if __name__ == "__main__":
    asyncio.run(main())

