"""
护工数据生成项目主程序
NDIS护工服务记录合成数据生成器
"""

import asyncio
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/generator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入项目模块
from config import get_config
from llm_data_generator import LLMDataGenerator
from free_llm_generator import FreeLLMDataGenerator
from data_validator import ComprehensiveValidator
from carer_data_schema import CarerServiceRecord


class CarersDataProject:
    """护工数据生成项目主类"""
    
    def __init__(self, config_override: Optional[dict] = None, free_mode: bool = False):
        """初始化项目"""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.free_mode = free_mode
        
        # 创建必要目录
        self._create_directories()
        
        # 初始化组件
        if free_mode:
            self.generator = FreeLLMDataGenerator(self.config)
            logger.info("护工数据生成项目初始化完成（免费模式）")
        else:
            self.generator = LLMDataGenerator(self.config)
            logger.info("护工数据生成项目初始化完成（标准模式）")
            
        self.validator = ComprehensiveValidator(self.config)
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            Path(self.config["output"]["output_dir"]),
            Path("logs"),
            Path("reports")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建目录: {directory}")
    
    async def generate_dataset(self, 
                             size: int = 1000,
                             validate: bool = True,
                             save_formats: List[str] = None) -> dict:
        """生成数据集并进行验证"""
        
        logger.info(f"开始生成 {size} 条护工服务记录")
        start_time = datetime.now()
        
        try:
            # 生成数据
            records = await self.generator.generate_dataset(size)
            
            if not records:
                raise ValueError("未能生成任何有效记录")
            
            logger.info(f"成功生成 {len(records)} 条记录")
            
            # 保存数据
            save_formats = save_formats or ["json", "csv", "jsonl"]
            saved_files = self.generator.save_dataset(records, "carers_synthetic_data")
            
            result = {
                "success": True,
                "generated_records": len(records),
                "target_size": size,
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "saved_files": saved_files,
                "validation_results": None
            }
            
            # 数据验证
            if validate and records:
                logger.info("开始数据验证...")
                validation_start = datetime.now()
                
                validation_results = self.validator.comprehensive_validation(records)
                validation_time = (datetime.now() - validation_start).total_seconds()
                
                # 保存验证报告
                report_file = self.validator.save_validation_report(
                    validation_results, 
                    f"validation_report_{len(records)}records.json"
                )
                
                result["validation_results"] = validation_results
                result["validation_time"] = validation_time
                result["validation_report"] = report_file
                
                logger.info(f"数据验证完成，总体评分: {validation_results.get('overall_score', 0)}")
            
            # 生成总结报告
            summary_report = self._generate_summary_report(result)
            result["summary_report"] = summary_report
            
            logger.info(f"数据生成项目完成，总用时: {result['generation_time']:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"数据生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _generate_summary_report(self, result: dict) -> str:
        """生成总结报告"""
        report_lines = [
            "# NDIS护工数据生成项目总结报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 数据生成结果",
            f"- 目标记录数: {result.get('target_size', 0)}",
            f"- 实际生成: {result.get('generated_records', 0)}",
            f"- 成功率: {(result.get('generated_records', 0) / result.get('target_size', 1) * 100):.1f}%",
            f"- 生成用时: {result.get('generation_time', 0):.2f}秒",
            ""
        ]
        
        # 添加验证结果
        if result.get("validation_results"):
            validation = result["validation_results"]
            report_lines.extend([
                "## 数据质量验证",
                f"- 总体评分: {validation.get('overall_score', 0)}/100",
                f"- 隐私评分: {validation.get('privacy_analysis', {}).get('anonymization_score', 0)}/100",
                f"- 真实性评分: {validation.get('utility_analysis', {}).get('realism_score', 0)}/100",
                f"- 验证用时: {result.get('validation_time', 0):.2f}秒",
                ""
            ])
            
            # 添加建议
            recommendations = validation.get("recommendations", [])
            if recommendations:
                report_lines.extend([
                    "## 改进建议",
                    *[f"- {rec}" for rec in recommendations],
                    ""
                ])
        
        # 添加文件信息
        saved_files = result.get("saved_files", {})
        if saved_files:
            report_lines.extend([
                "## 生成的文件",
                *[f"- {format_type}: {filepath}" for format_type, filepath in saved_files.items()],
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"summary_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"总结报告已保存: {report_file}")
        return str(report_file)
    
    async def validate_existing_data(self, data_file: str) -> dict:
        """验证现有数据文件"""
        logger.info(f"验证现有数据文件: {data_file}")
        
        try:
            # 加载数据
            records = self._load_data_file(data_file)
            
            # 执行验证
            validation_results = self.validator.comprehensive_validation(records)
            
            # 保存验证报告
            report_file = self.validator.save_validation_report(
                validation_results,
                f"validation_existing_{Path(data_file).stem}.json"
            )
            
            return {
                "success": True,
                "validated_records": len(records),
                "validation_results": validation_results,
                "validation_report": report_file
            }
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _load_data_file(self, filepath: str) -> List[CarerServiceRecord]:
        """加载数据文件"""
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        if file_path.suffix.lower() == '.json':
            return self._load_json_data(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            return self._load_jsonl_data(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv_data(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _load_json_data(self, file_path: Path) -> List[CarerServiceRecord]:
        """加载JSON数据"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = []
        for item in data:
            try:
                # 这里需要从字典转换回对象，简化处理
                record = CarerServiceRecord(**item)
                records.append(record)
            except Exception as e:
                logger.warning(f"跳过无效记录: {e}")
        
        return records
    
    def _load_jsonl_data(self, file_path: Path) -> List[CarerServiceRecord]:
        """加载JSONL数据"""
        import json
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    record = CarerServiceRecord(**item)
                    records.append(record)
                except Exception as e:
                    logger.warning(f"跳过无效记录: {e}")
        
        return records
    
    def _load_csv_data(self, file_path: Path) -> List[CarerServiceRecord]:
        """加载CSV数据"""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        records = []
        
        for _, row in df.iterrows():
            try:
                # 简化的CSV到对象转换
                item = row.to_dict()
                record = CarerServiceRecord(**item)
                records.append(record)
            except Exception as e:
                logger.warning(f"跳过无效记录: {e}")
        
        return records


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NDIS护工数据生成器")
    parser.add_argument("--size", type=int, default=1000, help="生成记录数量")
    parser.add_argument("--no-validate", action="store_true", help="跳过数据验证")
    parser.add_argument("--validate-file", type=str, help="验证现有数据文件")
    parser.add_argument("--output-formats", nargs="+", default=["json", "csv", "jsonl"], 
                       help="输出格式")
    parser.add_argument("--config", type=str, help="自定义配置文件")
    parser.add_argument("--free-mode", action="store_true", help="使用免费LLM模式")
    parser.add_argument("--check-free-services", action="store_true", help="检查可用的免费服务")
    
    args = parser.parse_args()
    
    try:
        # 检查免费服务状态
        if args.check_free_services:
            from free_config import get_setup_instructions, check_available_services
            print(get_setup_instructions())
            available = check_available_services()
            print(f"\n📊 服务状态详情: {available}")
            return
        
        # 初始化项目
        project = CarersDataProject(free_mode=args.free_mode)
        
        if args.validate_file:
            # 验证现有文件
            result = await project.validate_existing_data(args.validate_file)
        else:
            # 生成新数据
            result = await project.generate_dataset(
                size=args.size,
                validate=not args.no_validate,
                save_formats=args.output_formats
            )
        
        # 输出结果
        if result["success"]:
            print("\n✅ 项目执行成功!")
            if "generated_records" in result:
                print(f"📊 生成记录数: {result['generated_records']}")
            if "validation_results" in result and result["validation_results"]:
                print(f"🎯 质量评分: {result['validation_results']['overall_score']}/100")
            if "saved_files" in result:
                print("📁 生成的文件:")
                for format_type, filepath in result["saved_files"].items():
                    print(f"   {format_type}: {filepath}")
        else:
            print(f"\n❌ 项目执行失败: {result.get('error', '未知错误')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        logger.exception("主程序异常")
        sys.exit(1)


if __name__ == "__main__":
    # 确保事件循环在Windows上正常工作
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
