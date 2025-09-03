"""
项目初始化脚本
创建必要的目录结构并进行基础测试
"""

import os
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """创建项目目录结构"""
    directories = [
        "output",
        "logs", 
        "reports",
        "templates"
    ]
    
    base_dir = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"创建目录: {dir_path}")

def test_imports():
    """测试主要模块导入"""
    try:
        logger.info("测试模块导入...")
        
        from carer_data_schema import CarerServiceRecord, ServiceType
        logger.info("✅ carer_data_schema 导入成功")
        
        from config import get_config
        logger.info("✅ config 导入成功")
        
        from data_validator import ComprehensiveValidator
        logger.info("✅ data_validator 导入成功")
        
        # 测试演示生成器（不需要LLM）
        from demo_generator import DemoDataGenerator
        logger.info("✅ demo_generator 导入成功")
        
        logger.info("所有核心模块导入成功!")
        return True
        
    except Exception as e:
        logger.error(f"模块导入失败: {e}")
        return False

def run_basic_test():
    """运行基础功能测试"""
    try:
        logger.info("开始基础功能测试...")
        
        # 测试配置
        from config import get_config
        config = get_config()
        logger.info(f"配置加载成功, 项目名称: {config['project']['name']}")
        
        # 测试数据模式
        from carer_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome
        from datetime import date
        
        test_record = CarerServiceRecord(
            record_id="SR12345678",
            carer_id="CR123456",
            participant_id="PT654321", 
            service_date=date.today(),
            service_type=ServiceType.PERSONAL_CARE,
            duration_hours=2.5,
            narrative_notes="测试记录：为参与者提供个人护理服务，协助完成日常生活活动。参与者配合度良好，积极参与各项护理活动，达到了预期的护理目标。整个服务过程顺利进行。"
        )
        
        logger.info("✅ 数据模式测试成功")
        
        # 测试验证器
        from data_validator import ComprehensiveValidator
        validator = ComprehensiveValidator()
        
        validation_result = validator.comprehensive_validation([test_record])
        logger.info(f"✅ 验证器测试成功, 总体评分: {validation_result['overall_score']}")
        
        logger.info("基础功能测试全部通过!")
        return True
        
    except Exception as e:
        logger.error(f"基础功能测试失败: {e}")
        return False

def run_demo_generation():
    """运行演示数据生成"""
    try:
        logger.info("开始演示数据生成测试...")
        
        from demo_generator import DemoDataGenerator
        
        generator = DemoDataGenerator()
        
        # 生成小批量测试数据
        test_size = 10
        records = generator.generate_dataset(test_size)
        
        if records and len(records) == test_size:
            logger.info(f"✅ 成功生成 {len(records)} 条演示记录")
            
            # 保存数据
            saved_files = generator.save_dataset(records, "test_demo_data")
            logger.info(f"✅ 数据保存成功: {list(saved_files.keys())}")
            
            # 运行验证
            validation_results = generator.validator.comprehensive_validation(records)
            logger.info(f"✅ 验证完成, 质量评分: {validation_results['overall_score']}")
            
            # 保存验证报告
            report_file = generator.validator.save_validation_report(
                validation_results, 
                "test_validation_report.json"
            )
            logger.info(f"✅ 验证报告保存: {report_file}")
            
            return True
        else:
            logger.error(f"数据生成失败，期望 {test_size} 条，实际 {len(records) if records else 0} 条")
            return False
            
    except Exception as e:
        logger.error(f"演示数据生成测试失败: {e}")
        logger.exception("详细错误信息:")
        return False

def main():
    """主函数"""
    logger.info("开始项目初始化和测试...")
    
    # 1. 创建目录
    create_directories()
    
    # 2. 测试导入
    if not test_imports():
        logger.error("模块导入测试失败，请检查依赖安装")
        sys.exit(1)
    
    # 3. 基础功能测试
    if not run_basic_test():
        logger.error("基础功能测试失败")
        sys.exit(1)
    
    # 4. 演示数据生成测试
    if not run_demo_generation():
        logger.error("演示数据生成测试失败")
        sys.exit(1)
    
    logger.info("🎉 项目初始化和测试全部成功!")
    logger.info("项目已准备就绪，可以开始使用以下命令:")
    logger.info("  python demo_generator.py  # 运行演示数据生成")
    logger.info("  python main.py --size 100  # 运行完整数据生成")

if __name__ == "__main__":
    main()
