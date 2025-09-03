#!/usr/bin/env python3
"""
Quick Dashboard Test
快速验证仪表板核心功能
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from config import get_dashboard_config
        print("✅ 配置模块导入成功")
        
        config = get_dashboard_config()
        print(f"✅ 配置加载成功: {config['dashboard']['title']}")
        
    except Exception as e:
        print(f"❌ 配置模块错误: {e}")
        return False
    
    try:
        from data_aggregator import DataAggregator
        aggregator = DataAggregator()
        print("✅ 数据聚合器初始化成功")
        
    except Exception as e:
        print(f"❌ 数据聚合器错误: {e}")
        return False
    
    try:
        from demo import DashboardDemo
        demo = DashboardDemo()
        print("✅ 演示数据生成器初始化成功")
        
    except Exception as e:
        print(f"❌ 演示数据生成器错误: {e}")
        return False
    
    return True

def test_dependencies():
    """测试依赖包"""
    print("\n📦 测试依赖包...")
    
    required_packages = ['streamlit', 'plotly', 'pandas', 'flask']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 缺失")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ 缺失的包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")
    
    dashboard_dir = Path(__file__).parent
    required_files = [
        "config.py",
        "data_aggregator.py", 
        "streamlit_app.py",
        "run_dashboard.py",
        "demo.py",
        "requirements.txt"
    ]
    
    missing = []
    for file_name in required_files:
        if (dashboard_dir / file_name).exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 缺失")
            missing.append(file_name)
    
    # 检查数据目录
    data_dir = dashboard_dir / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ 创建数据目录")
    else:
        print("✅ 数据目录存在")
    
    return len(missing) == 0

def test_demo_data():
    """测试演示数据生成"""
    print("\n🎭 测试演示数据生成...")
    
    try:
        from demo import DashboardDemo
        demo = DashboardDemo()
        
        # 生成少量测试数据
        records = demo.generate_demo_data(3)
        print(f"✅ 成功生成 {len(records)} 条测试记录")
        
        # 生成验证报告
        validation_report = demo.generate_validation_report(records)
        print(f"✅ 生成验证报告，评分: {validation_report['overall_score']:.1f}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示数据生成错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 NDIS 仪表板快速测试")
    print("=" * 40)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖包", test_dependencies), 
        ("文件结构", test_file_structure),
        ("演示数据", test_demo_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # 空行分隔
        except Exception as e:
            print(f"💥 {test_name}测试异常: {e}\n")
    
    print("=" * 40)
    print(f"🏁 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！仪表板已准备就绪。")
        print("\n下一步:")
        print("1. 生成演示数据: python demo.py --records 100")
        print("2. 启动仪表板: python run_dashboard.py")
        print("3. 打开浏览器: http://localhost:8501")
        return True
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)



