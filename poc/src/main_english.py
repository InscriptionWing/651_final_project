"""
英文版主程序
专门用于生成英文NDIS护工数据
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

from pure_llm_english_generator import PureLLMEnglishGenerator
from english_data_schema import CarerServiceRecord, EnglishDataValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnglishCarersDataProject:
    """英文护工数据生成项目 - 纯LLM版本"""
    
    def __init__(self):
        """初始化项目"""
        self.generator = PureLLMEnglishGenerator()
        self.validator = EnglishDataValidator()
        
        # 创建输出目录
        self._create_directories()
        
        logger.info("English Carers Data Project initialized successfully (Pure LLM mode)")
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = ["output", "logs", "reports"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def generate_dataset(self, 
                             size: int = 100, 
                             validate: bool = True,
                             save_formats: List[str] = None) -> Dict:
        """生成英文数据集"""
        
        if save_formats is None:
            save_formats = ["json", "csv", "jsonl"]
        
        logger.info(f"Starting English dataset generation: {size} records")
        
        try:
            # 生成数据
            records = await self.generator.generate_dataset(size)
            
            if not records:
                raise Exception("No valid records generated")
            
            logger.info(f"Successfully generated {len(records)} English records")
            
            # 数据验证
            validation_result = None
            if validate:
                logger.info("Performing data validation...")
                validation_result = self.validator.validate_data_quality(records)
                logger.info("Basic validation completed")
            
            # 保存数据集
            saved_files = {}
            if "json" in save_formats or "jsonl" in save_formats or "csv" in save_formats:
                saved_files = self.generator.save_dataset(records, "pure_llm_english_carers")
            
            # 构建结果
            result = {
                "success": True,
                "records_count": len(records),
                "saved_files": saved_files,
                "validation": validation_result,
                "sample_records": [record.to_dict() for record in records[:3]]  # 前3个作为样本
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "records_count": 0
            }
    
    async def validate_existing_data(self, file_path: str) -> Dict:
        """验证现有数据文件"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            # 转换为CarerServiceRecord对象
            records = []
            for item in data:
                try:
                    record = CarerServiceRecord(**item)
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Invalid record: {e}")
            
            # 执行验证
            validation_result = self.validator.validate_data_quality(records)
            
            return {
                "success": True,
                "records_count": len(records),
                "validation": validation_result
            }
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="English NDIS Carers Data Generator (Pure LLM)")
    parser.add_argument("--size", type=int, default=100, 
                       help="Number of records to generate (default: 100)")
    parser.add_argument("--no-validate", action="store_true", 
                       help="Skip data validation")
    parser.add_argument("--validate-file", type=str, 
                       help="Validate existing data file")
    parser.add_argument("--output-formats", nargs="+", 
                       default=["json", "csv", "jsonl"],
                       choices=["json", "csv", "jsonl"],
                       help="Output formats (default: json csv jsonl)")
    
    args = parser.parse_args()
    
    try:
        project = EnglishCarersDataProject()
        
        if args.validate_file:
            # 验证现有文件
            result = await project.validate_existing_data(args.validate_file)
            
            if result["success"]:
                print(f"\n✅ Data validation completed successfully!")
                print(f"📊 Records validated: {result['records_count']}")
                if result.get("validation"):
                    val = result["validation"]
                    print(f"📊 Total records: {val.get('total_records', 'N/A')}")
                    print(f"👥 Unique carers: {val.get('unique_carers', 'N/A')}")
                    print(f"🎯 Unique participants: {val.get('unique_participants', 'N/A')}")
            else:
                print(f"❌ Validation failed: {result['error']}")
        
        else:
            # 生成新数据集
            result = await project.generate_dataset(
                size=args.size,
                validate=not args.no_validate,
                save_formats=args.output_formats
            )
            
            if result["success"]:
                print(f"\n✅ English dataset generation completed successfully!")
                print(f"📊 Generated records: {result['records_count']}")
                
                if result.get("validation"):
                    val = result["validation"]
                    print(f"📊 Total records: {val.get('total_records', 'N/A')}")
                    print(f"👥 Unique carers: {val.get('unique_carers', 'N/A')}")
                    print(f"🎯 Unique participants: {val.get('unique_participants', 'N/A')}")
                    print(f"⏱️ Average duration: {val.get('avg_duration', 'N/A'):.2f} hours" if val.get('avg_duration') else "⏱️ Average duration: N/A")
                    print(f"📝 Average narrative length: {val.get('avg_narrative_length', 'N/A'):.0f} characters" if val.get('avg_narrative_length') else "📝 Average narrative length: N/A")
                
                print(f"📁 Saved files:")
                for format_type, filepath in result["saved_files"].items():
                    print(f"   {format_type}: {filepath}")
                
                # 显示样本记录信息
                if result.get("sample_records"):
                    sample = result["sample_records"][0]
                    print(f"\n📋 Sample record:")
                    print(f"   Carer: {sample.get('carer_name', 'N/A')}")
                    print(f"   Service Type: {sample['service_type']}")
                    print(f"   Duration: {sample['duration_hours']} hours")
                    print(f"   Outcome: {sample['service_outcome']}")
                    print(f"   Location: {sample['location_type']}")
                    print(f"   Narrative: {sample['narrative_notes'][:150]}...")
            
            else:
                print(f"❌ Generation failed: {result['error']}")
                
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
