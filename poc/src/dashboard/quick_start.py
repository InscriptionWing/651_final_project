#!/usr/bin/env python3
"""
仪表板快速启动脚本
一键安装、配置和启动NDIS护工数据流水线仪表板
"""

import sys
import subprocess
import os
from pathlib import Path
import time

def print_banner():
    """显示欢迎横幅"""
    print("=" * 60)
    print("🚀 NDIS 护工数据流水线仪表板 - 快速启动")
    print("   NDIS Carer Data Pipeline Dashboard - Quick Start")
    print("=" * 60)
    print()

def check_python():
    """检查Python版本"""
    print("🔍 检查Python环境...")
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """安装依赖包"""
    print("\n📦 安装依赖包...")
    
    required_packages = ['streamlit', 'plotly', 'flask', 'schedule']
    
    # 检查已安装的包
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"⏳ {package} 需要安装")
    
    if missing_packages:
        print(f"\n🔧 安装缺失的包: {' '.join(missing_packages)}")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ 依赖包安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
            print("请手动运行: pip install streamlit plotly flask schedule")
            return False
    else:
        print("✅ 所有依赖包已安装")
        return True

def generate_demo_data():
    """生成演示数据"""
    print("\n🎭 生成演示数据...")
    
    try:
        # 添加路径
        sys.path.append(str(Path(__file__).parent.parent))
        
        from demo import DashboardDemo
        demo = DashboardDemo()
        
        # 生成演示数据
        demo.run_demo(150)  # 生成150条记录
        print("✅ 演示数据生成完成")
        return True
        
    except Exception as e:
        print(f"❌ 演示数据生成失败: {e}")
        return False

def start_dashboard():
    """启动仪表板"""
    print("\n🚀 启动仪表板...")
    print("   仪表板将在浏览器中自动打开")
    print("   地址: http://localhost:8501")
    print("   按 Ctrl+C 停止仪表板")
    print()
    
    try:
        # 确保在正确的目录中
        dashboard_dir = Path(__file__).parent
        os.chdir(dashboard_dir)
        
        # 启动Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("⏳ 启动中...")
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n✅ 仪表板已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print_banner()
    
    # 步骤1: 检查Python环境
    if not check_python():
        input("按回车键退出...")
        return
    
    # 步骤2: 安装依赖
    if not install_dependencies():
        print("\n⚠️ 请先手动安装依赖包，然后重新运行此脚本")
        input("按回车键退出...")
        return
    
    # 步骤3: 生成演示数据
    print("\n" + "=" * 40)
    choice = input("是否生成新的演示数据? (y/N): ").lower().strip()
    
    if choice in ['y', 'yes', '是']:
        if not generate_demo_data():
            print("⚠️ 演示数据生成失败，但可以继续使用现有数据")
    else:
        print("✅ 跳过演示数据生成，使用现有数据")
    
    # 步骤4: 启动仪表板
    print("\n" + "=" * 40)
    print("🎉 准备启动仪表板!")
    print("\n功能预览:")
    print("  📊 实时KPI监控 - 通过率、吞吐量等关键指标")
    print("  🚦 质量门分析 - 验证失败原因和趋势")
    print("  📋 记录浏览器 - 搜索和检查个别记录")
    print("  🎯 数据分布 - 服务类型和结果分析")
    print("  📝 模板监控 - 模板使用和多样性")
    print("  📤 数据导出 - 多格式报告生成")
    
    input("\n按回车键启动仪表板...")
    
    start_dashboard()
    
    print("\n" + "=" * 60)
    print("🎊 感谢使用NDIS护工数据流水线仪表板!")
    print("   如有问题，请查看 README.md 或 SETUP_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见!")
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        input("按回车键退出...")
        sys.exit(1)
