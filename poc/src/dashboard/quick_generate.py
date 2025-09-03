#!/usr/bin/env python3
"""
快速数据生成和仪表板更新
专为快速演示设计，基于现有英文数据模式
"""

import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from english_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome, LocationType
    from data_aggregator import DataAggregator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

class QuickEnglishDataGenerator:
    """快速英文数据生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.carer_names = [
            "Emma Johnson", "Liam Wilson", "Olivia Brown", "Noah Davis", "Ava Miller",
            "William Garcia", "Sophia Rodriguez", "James Martinez", "Isabella Anderson", "Benjamin Taylor",
            "Mia Thomas", "Lucas Jackson", "Charlotte White", "Henry Harris", "Amelia Martin",
            "Alexander Lee", "Harper Thompson", "Michael Garcia", "Evelyn Lewis", "Daniel Walker"
        ]
        
        self.narrative_templates = {
            ServiceType.PERSONAL_CARE: [
                "{carer} provided personal care assistance to the participant, focusing on daily hygiene and grooming activities. The participant showed good cooperation and engagement throughout the {duration}-hour session at {location}. Techniques included verbal guidance and physical assistance to promote independence.",
                "{carer} delivered comprehensive personal care support during a {duration}-hour session. The participant demonstrated positive response to assistance with bathing, dressing, and mobility tasks. Professional care standards were maintained throughout the service at {location}.",
                "{carer} assisted the participant with personal care needs including grooming and hygiene support. The {duration}-hour session was conducted with sensitivity and respect for participant dignity. Adaptive techniques were employed to encourage independence at {location}."
            ],
            ServiceType.HOUSEHOLD_TASKS: [
                "{carer} supported the participant with household management tasks during a {duration}-hour session at {location}. Activities included meal preparation, cleaning, and organizing living spaces. The participant actively engaged in learning domestic skills and showed improvement in task completion.",
                "{carer} provided household task assistance focusing on kitchen activities and general cleaning. The {duration}-hour session at {location} emphasized skill development and independence. The participant demonstrated increased confidence in managing daily household responsibilities.",
                "{carer} facilitated household task training including laundry, meal planning, and basic maintenance activities. During the {duration}-hour session at {location}, the participant learned organizational strategies and time management techniques for domestic tasks."
            ],
            ServiceType.COMMUNITY_ACCESS: [
                "{carer} accompanied the participant on a community access outing to enhance social participation. The {duration}-hour session at {location} focused on building confidence in public settings and developing social interaction skills. The participant engaged well with community activities.",
                "{carer} facilitated community participation activities during a {duration}-hour session. At {location}, the participant practiced social skills and community navigation. Professional support was provided to ensure safe and meaningful community engagement.",
                "{carer} supported the participant in accessing community resources and social activities. The {duration}-hour session at {location} included skill development in public interaction and community participation. The participant showed positive engagement throughout."
            ],
            ServiceType.TRANSPORT: [
                "{carer} provided transport assistance and mobility support during a {duration}-hour session. The participant received guidance on safe travel practices and route planning to {location}. Professional support ensured safe and efficient transportation with educational components.",
                "{carer} delivered transport assistance focusing on independence and safety skills. The {duration}-hour session included travel training and mobility support to {location}. The participant demonstrated improved confidence in transportation planning and execution.",
                "{carer} facilitated safe transport and provided mobility assistance during the {duration}-hour session. Professional guidance was given on public transport use and travel safety to {location}. The participant actively participated in travel planning activities."
            ],
            ServiceType.SOCIAL_SUPPORT: [
                "{carer} provided social and emotional support during a {duration}-hour session at {location}. The focus was on building communication skills and emotional regulation techniques. The participant engaged positively and expressed appreciation for the supportive environment.",
                "{carer} delivered social support services emphasizing relationship building and communication development. The {duration}-hour session at {location} included conversation practice and social skill enhancement. The participant demonstrated improved social confidence.",
                "{carer} facilitated social interaction and emotional support activities during the {duration}-hour session. At {location}, the participant practiced social communication and received encouragement for personal development. Positive therapeutic rapport was maintained throughout."
            ]
        }
        
        self.support_techniques = [
            "Verbal guidance and encouragement",
            "Visual prompts and demonstrations", 
            "Physical assistance and modeling",
            "Environmental modifications",
            "Positive reinforcement strategies",
            "Task breakdown and sequencing",
            "Adaptive equipment utilization",
            "Sensory support techniques",
            "Communication aids and tools",
            "Routine establishment methods"
        ]
        
        self.challenges = [
            "Initial hesitation with new activities",
            "Communication barriers requiring patience",
            "Environmental distractions affecting focus",
            "Time management challenges",
            "Equipment accessibility issues",
            "Weather-related modifications needed",
            "Scheduling coordination difficulties",
            "Technical equipment malfunctions",
            "Physical fatigue during longer sessions",
            "Emotional regulation support needed"
        ]
    
    def generate_record(self) -> CarerServiceRecord:
        """生成单条英文护工服务记录"""
        # 随机选择基本信息
        carer_name = random.choice(self.carer_names)
        service_type = random.choice(list(ServiceType))
        location_type = random.choice(list(LocationType))
        service_outcome = random.choices(
            list(ServiceOutcome),
            weights=[0.6, 0.25, 0.1, 0.05]  # positive, neutral, negative, incomplete
        )[0]
        
        # 生成时长
        duration = round(random.uniform(0.5, 6.0), 2)
        
        # 生成服务日期（过去30天内）
        days_back = random.randint(0, 30)
        service_date = date.today() - timedelta(days=days_back)
        
        # 生成叙述内容
        narrative_template = random.choice(self.narrative_templates[service_type])
        narrative = narrative_template.format(
            carer=carer_name,
            duration=duration,
            location=location_type.value
        )
        
        # 随机添加结果描述
        if service_outcome == ServiceOutcome.POSITIVE:
            narrative += " The session exceeded expectations with excellent participant engagement and goal achievement."
        elif service_outcome == ServiceOutcome.NEGATIVE:
            narrative += " Some challenges were encountered that limited the session's effectiveness."
        elif service_outcome == ServiceOutcome.INCOMPLETE:
            narrative += " The session was concluded early due to unforeseen circumstances."
        
        # 选择支持技术和挑战
        support_techniques = random.sample(self.support_techniques, random.randint(1, 3))
        challenges = random.sample(self.challenges, random.randint(0, 2))
        
        # 创建记录
        record = CarerServiceRecord(
            record_id=f"SR{random.randint(10000000, 99999999)}",
            carer_id=f"CR{random.randint(100000, 999999)}",
            participant_id=f"PT{random.randint(100000, 999999)}",
            service_date=service_date,
            service_type=service_type,
            duration_hours=duration,
            narrative_notes=narrative,
            carer_name=carer_name,
            location_type=location_type,
            location_details=f"{location_type.value} - Professional NDIS support environment",
            service_outcome=service_outcome,
            support_techniques_used=support_techniques,
            challenges_encountered=challenges,
            follow_up_required=random.choice([True, False]),
            billing_code=f"NDIS_{service_type.name}_{random.randint(1000, 9999)}"
        )
        
        return record
    
    def generate_dataset(self, size: int) -> List[CarerServiceRecord]:
        """生成数据集"""
        print(f"🤖 Generating {size} English carer service records...")
        
        records = []
        for i in range(size):
            try:
                record = self.generate_record()
                records.append(record)
                
                if (i + 1) % 10 == 0:
                    print(f"   Generated {i + 1}/{size} records...")
                    
            except Exception as e:
                print(f"   ⚠️ Error generating record {i+1}: {e}")
        
        print(f"✅ Successfully generated {len(records)} records")
        return records
    
    def save_dataset(self, records: List[CarerServiceRecord], filename_prefix: str = "quick_english"):
        """保存数据集"""
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{filename_prefix}_{timestamp}_{len(records)}records"
        
        # 转换为字典格式
        data = [record.to_dict() for record in records]
        
        # 保存JSON文件
        json_file = output_dir / f"{filename_base}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 保存JSONL文件
        jsonl_file = output_dir / f"{filename_base}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record_dict in data:
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        
        print(f"📁 Files saved:")
        print(f"   JSON: {json_file}")
        print(f"   JSONL: {jsonl_file}")
        
        return {
            "json": str(json_file),
            "jsonl": str(jsonl_file)
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick English Data Generator")
    parser.add_argument("--size", type=int, default=25, help="Number of records to generate")
    parser.add_argument("--update-dashboard", action="store_true", help="Update dashboard after generation")
    
    args = parser.parse_args()
    
    print("⚡ Quick English Data Generator")
    print("=" * 50)
    
    try:
        # 生成数据
        generator = QuickEnglishDataGenerator()
        records = generator.generate_dataset(args.size)
        
        if not records:
            print("❌ No records generated")
            return False
        
        # 保存数据
        saved_files = generator.save_dataset(records)
        
        # 更新仪表板（如果要求）
        if args.update_dashboard:
            print("\n📊 Updating dashboard...")
            aggregator = DataAggregator()
            result = aggregator.aggregate_all_data()
            
            if result.get("status") == "success":
                print("✅ Dashboard updated successfully!")
                output_metrics = result.get("output_metrics", {})
                print(f"   Total records in dashboard: {output_metrics.get('total_records', 0)}")
            else:
                print(f"❌ Dashboard update failed: {result.get('error', 'Unknown error')}")
        
        # 显示样本记录
        if records:
            sample = records[0]
            print(f"\n📋 Sample record:")
            print(f"   Carer: {sample.carer_name}")
            print(f"   Service: {sample.service_type.value}")
            print(f"   Duration: {sample.duration_hours} hours")
            print(f"   Location: {sample.location_type.value}")
            print(f"   Outcome: {sample.service_outcome.value}")
            print(f"   Narrative: {sample.narrative_notes[:100]}...")
        
        print(f"\n🎉 Quick generation completed!")
        print(f"   Generated: {len(records)} records")
        print(f"   Files: {len(saved_files)} formats")
        
        if args.update_dashboard:
            print(f"   Dashboard: Updated")
            print(f"\n🌐 View at: http://localhost:8501")
        else:
            print(f"\n💡 To update dashboard: python dashboard/data_aggregator.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
