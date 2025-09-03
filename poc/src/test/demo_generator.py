"""
演示数据生成器
用于快速生成小批量测试数据，无需LLM调用
"""

import json
import random
from datetime import datetime, date, timedelta
from typing import List, Dict
import logging
from pathlib import Path

from faker import Faker
from carer_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile
)
from config import get_config
from data_validator import ComprehensiveValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker
fake = Faker(['en_AU', 'zh_CN'])
fake.seed_instance(42)
random.seed(42)


class DemoDataGenerator:
    """演示数据生成器（不使用LLM）"""
    
    def __init__(self):
        self.config = get_config()
        self.validator = ComprehensiveValidator()
        
        # 预定义的叙述模板
        self.narrative_templates = self._load_narrative_templates()
        
        logger.info("演示数据生成器初始化完成")
    
    def _load_narrative_templates(self) -> Dict[str, List[str]]:
        """加载叙述模板"""
        templates = {
            "positive": [
                "为参与者{participant_name}提供{service_type}服务。参与者配合度很好，顺利完成了预定的护理目标。使用了{technique}方法，效果显著。服务在{location}进行，环境适宜。",
                "今日协助{participant_name}进行{service_type}。参与者表现积极，主动配合护理活动。采用了{technique}技术，获得了良好的反馈。整个过程顺利，达到了预期效果。",
                "为{participant_name}提供专业的{service_type}支持。参与者情绪稳定，能够积极参与各项活动。运用{technique}方法进行干预，取得了满意的成果。",
                "协助{participant_name}完成{service_type}任务。参与者展现出良好的合作态度，能够遵循指导完成相关活动。通过{technique}策略的运用，成功实现了护理目标。"
            ],
            "neutral": [
                "为参与者{participant_name}提供常规{service_type}服务。过程中参与者表现平稳，按照计划完成了基本护理项目。使用了标准的{technique}方法。",
                "今日为{participant_name}进行{service_type}。参与者状态稳定，能够配合完成必要的护理活动。采用了{technique}技术，效果一般。",
                "协助{participant_name}进行{service_type}。参与者情绪较为平稳，在护工引导下完成了相关任务。运用了{technique}方法。",
                "为{participant_name}提供{service_type}支持。参与者表现正常，能够参与大部分活动。通过{technique}策略进行干预。"
            ],
            "negative": [
                "为参与者{participant_name}提供{service_type}服务时遇到挑战。参与者情绪波动较大，对某些活动表现出抗拒。尝试使用{technique}方法缓解，效果有限。需要调整护理策略。",
                "协助{participant_name}进行{service_type}时遇到困难。参与者今日状态不佳，配合度较低。虽然采用了{technique}技术，但进展缓慢，需要后续跟进。",
                "为{participant_name}提供{service_type}支持过程中出现问题。参与者表现出焦虑情绪，影响了服务的正常进行。运用{technique}方法进行安抚，效果一般。",
                "今日{participant_name}的{service_type}服务遇到挑战。参与者对环境变化敏感，情绪不稳定。虽然使用了{technique}策略，但仍需要额外的支持和关注。"
            ]
        }
        
        # 从模板文件加载（如果存在）
        template_file = Path("templates_enhanced.txt")
        if template_file.exists():
            try:
                enhanced_templates = self._parse_template_file(template_file)
                templates.update(enhanced_templates)
                logger.info(f"从文件加载了增强模板: {len(enhanced_templates)} 个类别")
            except Exception as e:
                logger.warning(f"加载模板文件失败: {e}")
        
        return templates
    
    def _parse_template_file(self, file_path: Path) -> Dict[str, List[str]]:
        """解析模板文件"""
        templates = {"positive": [], "neutral": [], "negative": []}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '[' in line and ']' in line:
                    # 解析格式：[outcome] narrative
                    try:
                        outcome = line[line.find('[')+1:line.find(']')]
                        narrative = line[line.find(']')+1:].strip()
                        
                        if outcome in templates and narrative:
                            # 转换为模板格式
                            template_narrative = self._convert_to_template(narrative)
                            templates[outcome].append(template_narrative)
                    except Exception as e:
                        logger.debug(f"解析模板行失败: {line}, 错误: {e}")
        
        return templates
    
    def _convert_to_template(self, narrative: str) -> str:
        """将具体叙述转换为模板格式"""
        # 简单的模板转换 - 将人名替换为占位符
        template = narrative
        
        # 常见的人名模式替换
        names = ["Mia", "Lucas", "Zara", "Noah", "Sofia", "Liam", "Aria", "Mateo", "Yuki", "Priya"]
        for name in names:
            if name in template:
                template = template.replace(name, "{participant_name}")
                break
        
        # 添加服务类型和技术占位符（如果不存在）
        if "{service_type}" not in template and "{technique}" not in template:
            template = template + " 使用了{technique}方法进行{service_type}。"
        
        return template
    
    def generate_carer_profiles(self, count: int = 20) -> List[CarerProfile]:
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
    
    def generate_participant_profiles(self, count: int = 50) -> List[ParticipantProfile]:
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
        """生成叙述内容"""
        
        # 选择模板
        outcome_key = outcome.value if outcome else "positive"
        templates = self.narrative_templates.get(outcome_key, self.narrative_templates["positive"])
        
        if not templates:
            templates = self.narrative_templates["positive"]
        
        template = random.choice(templates)
        
        # 填充模板
        participant_name = participant_name or fake.first_name()
        service_type_cn = service_type.value
        technique = random.choice([
            "渐进式引导", "正向强化", "结构化支持", "感官调节",
            "认知重构", "行为塑造", "环境适应", "沟通辅助"
        ])
        location = random.choice([
            "参与者家中", "社区中心", "康复训练室", "户外环境",
            "安静的房间", "熟悉的环境", "专用活动区域"
        ])
        
        narrative = template.format(
            participant_name=participant_name,
            service_type=service_type_cn,
            technique=technique,
            location=location
        )
        
        # 确保长度在合理范围内
        if len(narrative) < 50:
            narrative += " 整个服务过程顺利进行，参与者状态良好。"
        elif len(narrative) > 500:
            narrative = narrative[:497] + "..."
        
        return narrative
    
    def generate_service_record(self,
                              carer: CarerProfile,
                              participant: ParticipantProfile,
                              service_date: date = None) -> CarerServiceRecord:
        """生成单条服务记录"""
        
        # 生成基础数据
        record_id = f"SR{random.randint(10000000, 99999999):08d}"
        service_date = service_date or (date.today() - timedelta(days=random.randint(1, 90)))
        
        # 选择服务类型（基于权重）
        service_weights = self.config["service"]["service_types_weights"]
        service_types = list(ServiceType)
        weights = [service_weights.get(st.value, 0.1) for st in service_types]
        service_type = random.choices(service_types, weights=weights)[0]
        
        # 选择服务结果（基于权重）
        outcome_weights = self.config["service"]["outcome_weights"]
        outcomes = list(ServiceOutcome)
        outcome_weights_list = [outcome_weights.get(oc.value, 0.1) for oc in outcomes]
        service_outcome = random.choices(outcomes, weights=outcome_weights_list)[0]
        
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
            "行为强化", "感官支持", "时间管理", "社交技能训练"
        ], random.randint(1, 3))
        
        challenges = []
        if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
            challenges = random.sample([
                "参与者情绪波动", "环境噪音干扰", "沟通困难",
                "注意力分散", "身体不适", "设备问题"
            ], random.randint(1, 2))
        
        # 参与者反应
        participant_responses = {
            ServiceOutcome.POSITIVE: ["配合良好", "积极参与", "表现出兴趣"],
            ServiceOutcome.NEUTRAL: ["表现平稳", "基本配合", "无特殊反应"],
            ServiceOutcome.NEGATIVE: ["表现抗拒", "情绪不稳", "需要额外支持"],
            ServiceOutcome.INCOMPLETE: ["中途停止", "注意力不集中", "需要休息"]
        }
        participant_response = random.choice(participant_responses.get(service_outcome, ["无特殊反应"]))
        
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
            location_details=f"{location_type.value}的具体区域",
            service_outcome=service_outcome,
            support_techniques_used=support_techniques,
            challenges_encountered=challenges,
            participant_response=participant_response,
            follow_up_required=service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]
        )
        
        return record
    
    def generate_dataset(self, size: int = 1000) -> List[CarerServiceRecord]:
        """生成完整数据集"""
        logger.info(f"开始生成 {size} 条演示记录")
        
        # 生成档案
        carers = self.generate_carer_profiles(max(10, size // 20))
        participants = self.generate_participant_profiles(max(20, size // 10))
        
        records = []
        
        for i in range(size):
            # 随机选择护工和参与者
            carer = random.choice(carers)
            participant = random.choice(participants)
            
            try:
                record = self.generate_service_record(carer, participant)
                records.append(record)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已生成 {i + 1} 条记录")
                    
            except Exception as e:
                logger.warning(f"生成第 {i+1} 条记录失败: {e}")
        
        logger.info(f"演示数据集生成完成，共 {len(records)} 条记录")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], prefix: str = "demo_carers_data") -> Dict[str, str]:
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
        
        logger.info(f"数据集已保存: {saved_files}")
        return saved_files


def main():
    """演示主函数"""
    generator = DemoDataGenerator()
    
    # 生成小规模测试数据
    test_size = 100
    logger.info(f"生成演示数据集（{test_size}条记录）")
    
    records = generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 执行验证
        logger.info("执行数据验证...")
        validation_results = generator.validator.comprehensive_validation(records)
        
        # 保存验证报告
        report_file = generator.validator.save_validation_report(
            validation_results, 
            f"demo_validation_report_{test_size}records.json"
        )
        
        # 输出结果
        print(f"\n✅ 演示数据生成完成!")
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
        logger.error("未能生成任何记录")


if __name__ == "__main__":
    main()

