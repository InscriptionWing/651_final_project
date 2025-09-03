"""
免费LLM数据生成器
使用免费的本地和在线LLM服务生成护工数据
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
import requests
import time
from dataclasses import asdict

from faker import Faker
from tenacity import retry, stop_after_attempt, wait_random_exponential

from carer_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, DataValidator
)
from config import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Faker
fake = Faker(['en_AU', 'zh_CN'])


class FreeLLMDataGenerator:
    """免费LLM驱动的护工数据生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化生成器"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # 设置随机种子
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # 初始化免费LLM客户端选项
        self.llm_options = {
            "ollama": self._init_ollama_client,
            "huggingface": self._init_huggingface_client,
            "template": self._init_template_generator,
            "rules": self._init_rules_generator
        }
        
        # 选择可用的LLM方法
        self.active_llm = self._select_available_llm()
        
        # 初始化ollama模型名称为默认值（会在init client时更新）
        self.ollama_model = "llama2"
        
        # 加载模板
        self.templates = self._load_templates()
        
        # 初始化数据验证器
        self.validator = DataValidator()
        
        # 预生成的护工和参与者档案
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info(f"免费LLM数据生成器初始化完成，使用方法: {self.active_llm}")
    
    def _init_ollama_client(self):
        """初始化Ollama本地客户端"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    # 保存检测到的模型，优先使用第一个
                    self.ollama_model = models[0]['name']
                    logger.info(f"检测到Ollama模型: {[m['name'] for m in models]}, 将使用: {self.ollama_model}")
                    return "ollama"
        except:
            pass
        return None
    
    def _init_huggingface_client(self):
        """初始化Hugging Face免费API客户端"""
        # 这里可以配置免费的Hugging Face Inference API
        # 需要用户注册免费账号并获取token
        hf_token = self.config.get("huggingface", {}).get("token")
        if hf_token and hf_token != "your_huggingface_token_here":
            return "huggingface"
        return None
    
    def _init_template_generator(self):
        """初始化基于模板的生成器"""
        return "template"
    
    def _init_rules_generator(self):
        """初始化基于规则的生成器"""
        return "rules"
    
    def _select_available_llm(self) -> str:
        """选择可用的LLM方法"""
        # 按优先级尝试不同的方法
        priority_order = ["ollama", "huggingface", "template", "rules"]
        
        for method in priority_order:
            if method in self.llm_options:
                result = self.llm_options[method]()
                if result:
                    return result
        
        # 默认使用基于规则的生成器
        return "rules"
    
    def _load_templates(self) -> List[str]:
        """加载增强模板"""
        template_file = Path("templates_enhanced.txt")
        if not template_file.exists():
            logger.warning(f"模板文件不存在: {template_file}")
            return self._get_default_templates()
        
        templates = []
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.strip().split('\n'):
                if line.strip() and '[' in line and ']' in line:
                    templates.append(line.strip())
        
        logger.info(f"加载了 {len(templates)} 个模板")
        return templates if templates else self._get_default_templates()
    
    def _get_default_templates(self) -> List[str]:
        """获取默认模板"""
        return [
            "[positive] 为参与者提供{service_type}服务。参与者配合度良好，积极参与各项活动。护工使用了{technique}方法，取得了满意的效果。整个服务过程顺利进行。",
            "[neutral] 为参与者提供{service_type}服务。参与者表现平稳，按照计划完成了基本活动。护工采用了{technique}技术，效果一般。",
            "[negative] 为参与者提供{service_type}服务时遇到挑战。参与者情绪波动，对某些活动表现出抗拒。护工尝试使用{technique}方法缓解，需要后续跟进。",
            "[positive] 护工协助参与者进行{service_type}活动。参与者表现积极，主动配合各项支持措施。通过{technique}策略的实施，达到了预期的护理目标。",
            "[neutral] 护工为参与者提供{service_type}支持。参与者状态稳定，能够配合完成必要的护理活动。采用{technique}方法进行干预。",
            "[negative] 护工在为参与者提供{service_type}服务过程中遇到困难。参与者需要额外的耐心和支持。虽然使用了{technique}技术，但仍需要持续关注。"
        ]
    
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    async def _call_ollama(self, prompt: str) -> str:
        """调用Ollama本地模型"""
        try:
            # 使用检测到的模型名称，如果没有则使用默认值
            model_name = getattr(self, 'ollama_model', 'llama2')
            
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API错误: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    async def _call_huggingface(self, prompt: str) -> str:
        """调用Hugging Face免费API"""
        try:
            # 使用免费的文本生成模型
            api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            headers = {
                "Authorization": f"Bearer {self.config.get('huggingface', {}).get('token', '')}",
                "Content-Type": "application/json"
            }
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return ""
            else:
                raise Exception(f"Hugging Face API错误: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Hugging Face调用失败: {e}")
            raise
    
    def _generate_with_template(self, 
                               service_type: ServiceType,
                               participant_profile: ParticipantProfile,
                               carer_profile: CarerProfile,
                               outcome: ServiceOutcome) -> Dict[str, Any]:
        """使用模板生成数据"""
        
        # 选择合适的模板
        outcome_templates = [t for t in self.templates if f"[{outcome.value}]" in t]
        if not outcome_templates:
            outcome_templates = [t for t in self.templates if "[positive]" in t]
        
        template = random.choice(outcome_templates)
        
        # 提取模板内容
        if ']' in template:
            narrative_template = template[template.find(']')+1:].strip()
        else:
            narrative_template = template
        
        # 填充模板变量
        techniques = [
            "渐进式引导", "正向强化", "结构化支持", "感官调节",
            "认知重构", "行为塑造", "环境适应", "沟通辅助"
        ]
        
        locations = [
            "参与者家中客厅", "社区中心活动室", "康复训练室", 
            "户外花园区域", "安静的阅读角", "专用治疗室"
        ]
        
        # 生成具体的叙述
        try:
            narrative = narrative_template.format(
                service_type=service_type.value,
                technique=random.choice(techniques),
                participant_name=fake.first_name(),
                location=random.choice(locations)
            )
        except KeyError:
            # 如果模板格式不匹配，使用简化版本
            narrative = f"为参与者提供{service_type.value}服务。护工采用{random.choice(techniques)}方法，在{random.choice(locations)}进行支持活动。"
        
        # 确保叙述长度合适
        if len(narrative) < 50:
            narrative += f" 护工采用专业的{random.choice(techniques)}策略，确保服务质量。"
        elif len(narrative) > 500:
            narrative = narrative[:497] + "..."
        
        # 生成其他字段
        support_techniques = random.sample(techniques, random.randint(1, 3))
        
        challenges = []
        if outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
            challenge_options = [
                "参与者情绪波动", "环境噪音干扰", "沟通困难",
                "注意力分散", "身体不适", "时间压力"
            ]
            challenges = random.sample(challenge_options, random.randint(1, 2))
        
        participant_responses = {
            ServiceOutcome.POSITIVE: ["积极配合", "表现出兴趣", "主动参与"],
            ServiceOutcome.NEUTRAL: ["基本配合", "表现平稳", "无特殊反应"],
            ServiceOutcome.NEGATIVE: ["表现抗拒", "情绪不稳", "需要额外支持"],
            ServiceOutcome.INCOMPLETE: ["中途停止", "注意力不集中", "需要休息"]
        }
        
        return {
            "narrative_notes": narrative,
            "service_outcome": outcome.value,
            "location_details": random.choice(locations),
            "support_techniques_used": support_techniques,
            "challenges_encountered": challenges,
            "participant_response": random.choice(participant_responses.get(outcome, ["无特殊反应"])),
            "follow_up_required": outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]
        }
    
    def _generate_with_rules(self,
                            service_type: ServiceType,
                            participant_profile: ParticipantProfile,
                            carer_profile: CarerProfile,
                            outcome: ServiceOutcome) -> Dict[str, Any]:
        """使用规则生成数据"""
        
        # 基于服务类型的叙述模板
        service_narratives = {
            ServiceType.PERSONAL_CARE: [
                "协助参与者进行个人卫生护理，包括洗漱、穿衣等日常活动。",
                "为参与者提供个人护理支持，确保其个人卫生和舒适度。",
                "在护工指导下，参与者完成了个人护理例行程序。"
            ],
            ServiceType.HOUSEHOLD_TASKS: [
                "协助参与者完成家务整理，包括清洁和物品归置。",
                "支持参与者参与家庭维护活动，培养生活技能。",
                "指导参与者进行基础家务管理，提高独立生活能力。"
            ],
            ServiceType.COMMUNITY_ACCESS: [
                "陪同参与者参与社区活动，促进社会融入。",
                "支持参与者在社区环境中的活动参与和互动。",
                "协助参与者适应社区环境，建立社会联系。"
            ],
            ServiceType.TRANSPORT: [
                "为参与者提供交通协助，确保安全到达目的地。",
                "陪同参与者进行必要的出行，提供途中支持。",
                "协助参与者使用公共交通或安排专车服务。"
            ]
        }
        
        # 选择基础叙述
        base_narratives = service_narratives.get(service_type, [
            f"为参与者提供{service_type.value}服务支持。"
        ])
        base_narrative = random.choice(base_narratives)
        
        # 根据结果添加具体描述
        outcome_descriptions = {
            ServiceOutcome.POSITIVE: [
                "参与者积极配合，顺利完成了所有计划活动。",
                "整个过程进行顺利，参与者表现出良好的参与度。",
                "参与者反应积极，达到了预期的服务目标。"
            ],
            ServiceOutcome.NEUTRAL: [
                "参与者表现平稳，按计划完成了基本活动。",
                "服务过程正常，参与者配合度一般。",
                "活动按既定计划进行，无特殊情况。"
            ],
            ServiceOutcome.NEGATIVE: [
                "参与者情绪波动较大，需要额外的耐心和支持。",
                "遇到一些挑战，参与者对某些活动表现出抗拒。",
                "服务过程中出现困难，需要调整策略。"
            ],
            ServiceOutcome.INCOMPLETE: [
                "由于参与者状态问题，活动未能完全完成。",
                "服务过程中断，需要重新安排时间。",
                "参与者需要休息，活动提前结束。"
            ]
        }
        
        outcome_desc = random.choice(outcome_descriptions.get(outcome, ["活动正常进行。"]))
        
        # 添加技术描述
        techniques = [
            "采用渐进式引导方法", "使用正向强化策略", "运用结构化支持技术",
            "实施感官调节方案", "应用认知重构技巧", "采用环境适应策略"
        ]
        technique_desc = random.choice(techniques)
        
        # 组合完整叙述
        full_narrative = f"{base_narrative} {outcome_desc} {technique_desc}，确保服务质量和参与者舒适度。"
        
        # 生成其他字段
        support_techniques = random.sample([
            "视觉提示", "口语指导", "物理协助", "环境调整",
            "行为强化", "感官支持", "时间管理", "社交技能训练"
        ], random.randint(1, 3))
        
        challenges = []
        if outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
            challenges = random.sample([
                "参与者情绪管理", "环境适应困难", "沟通障碍",
                "注意力维持", "体力限制", "时间安排冲突"
            ], random.randint(1, 2))
        
        locations = [
            f"{random.choice(['参与者家中', '社区中心', '康复中心', '户外场所'])}的{random.choice(['客厅', '活动室', '训练区', '安静角落'])}"
        ]
        
        participant_responses = {
            ServiceOutcome.POSITIVE: ["积极配合", "主动参与", "表现出色"],
            ServiceOutcome.NEUTRAL: ["基本配合", "表现稳定", "正常参与"],
            ServiceOutcome.NEGATIVE: ["需要鼓励", "情绪波动", "需要支持"],
            ServiceOutcome.INCOMPLETE: ["需要休息", "注意力分散", "状态不佳"]
        }
        
        return {
            "narrative_notes": full_narrative,
            "service_outcome": outcome.value,
            "location_details": random.choice(locations),
            "support_techniques_used": support_techniques,
            "challenges_encountered": challenges,
            "participant_response": random.choice(participant_responses.get(outcome, ["正常反应"])),
            "follow_up_required": outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """调用选定的LLM方法"""
        if self.active_llm == "ollama":
            return await self._call_ollama(prompt)
        elif self.active_llm == "huggingface":
            return await self._call_huggingface(prompt)
        else:
            # 对于template和rules方法，我们不需要调用外部LLM
            return ""
    
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
            
            # 根据选定的方法生成内容
            if self.active_llm in ["template", "rules"]:
                if self.active_llm == "template":
                    llm_data = self._generate_with_template(
                        service_type, participant, carer, service_outcome
                    )
                else:  # rules
                    llm_data = self._generate_with_rules(
                        service_type, participant, carer, service_outcome
                    )
            else:
                # 使用外部LLM
                prompt = self._build_generation_prompt(service_type, participant, carer)
                llm_response = await self._call_llm(prompt)
                
                try:
                    llm_data = json.loads(llm_response)
                except:
                    # 如果LLM响应解析失败，回退到规则生成
                    llm_data = self._generate_with_rules(
                        service_type, participant, carer, service_outcome
                    )
            
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
                service_outcome=ServiceOutcome(llm_data.get("service_outcome", service_outcome.value)),
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
    
    def _build_generation_prompt(self, 
                                service_type: ServiceType,
                                participant_profile: ParticipantProfile,
                                carer_profile: CarerProfile) -> str:
        """构建数据生成提示（用于外部LLM）"""
        
        prompt = f"""
生成一个护工服务记录的JSON数据，要求如下：

参与者信息：
- 年龄组: {participant_profile.age_group}
- 残疾类型: {participant_profile.disability_type}
- 支持级别: {participant_profile.support_level}

护工信息：
- 认证级别: {carer_profile.certification_level}
- 经验年限: {carer_profile.years_experience}年
- 专业领域: {', '.join(carer_profile.specializations)}

服务类型: {service_type.value}

请生成JSON格式：
{{
  "narrative_notes": "详细的服务记录叙述（100-300字符）",
  "service_outcome": "positive/neutral/negative/incomplete之一",
  "location_details": "具体服务地点描述",
  "support_techniques_used": ["使用的支持技术列表"],
  "challenges_encountered": ["遇到的挑战列表"],
  "participant_response": "参与者反应描述",
  "follow_up_required": true/false
}}
"""
        
        return prompt
    
    async def generate_dataset(self, size: int = 1000) -> List[CarerServiceRecord]:
        """生成完整数据集"""
        logger.info(f"开始生成 {size} 条服务记录（使用{self.active_llm}方法）")
        
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
                
                # 创建任务
                task = self.generate_service_record(
                    carer, participant, service_date, service_type
                )
                batch_tasks.append(task)
            
            # 执行批量任务
            if self.active_llm in ["template", "rules"]:
                # 同步执行，因为不需要外部API调用
                batch_results = []
                for task in batch_tasks:
                    try:
                        result = await task
                        batch_results.append(result)
                    except Exception as e:
                        batch_results.append(e)
            else:
                # 异步执行
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
                    filename_prefix: str = "free_llm_carers_data") -> Dict[str, str]:
        """保存数据集到多种格式"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
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
    """主函数 - 演示免费LLM数据生成"""
    generator = FreeLLMDataGenerator()
    
    # 生成测试数据
    test_size = 50
    logger.info(f"生成测试数据集（{test_size}条记录）")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # 保存数据
        saved_files = generator.save_dataset(records, "free_llm_test_data")
        
        # 打印质量报告
        quality_report = generator.validator.validate_data_quality(records)
        logger.info(f"数据质量报告: {quality_report}")
        
        # 显示示例记录
        if records:
            logger.info("示例记录:")
            print(records[0].to_json())
            
        print(f"\n✅ 免费LLM数据生成完成!")
        print(f"📊 生成记录数: {len(records)}")
        print(f"🔧 使用方法: {generator.active_llm}")
        print(f"📁 保存的文件:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
    else:
        logger.error("未能生成任何有效记录")


if __name__ == "__main__":
    asyncio.run(main())
