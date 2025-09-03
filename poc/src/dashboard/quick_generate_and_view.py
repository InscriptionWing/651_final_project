#!/usr/bin/env python3
"""
🚀 一键生成数据并查看仪表板
最简化的数据生成和仪表板更新流程
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """打印横幅"""
    print("\n" + "=" * 60)
    print("🚀 一键生成英文护工数据并查看仪表板")
    print("   Generate English Carer Data & View Dashboard")
    print("=" * 60)

def run_command(cmd, description, timeout=120, capture_output=True):
    """运行命令并处理结果"""
    print(f"\n⚡ {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                print(f"✅ {description} - 成功完成!")
                return True, result.stdout
            else:
                print(f"❌ {description} - 失败!")
                print(f"错误信息: {result.stderr[:200]}...")
                return False, result.stderr
        else:
            # 直接运行，不捕获输出（用于启动仪表板）
            subprocess.run(cmd)
            return True, ""
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - 超时 ({timeout}秒)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False, str(e)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="一键生成数据并查看仪表板")
    parser.add_argument("--size", type=int, default=20, help="生成记录数量 (默认: 20)")
    parser.add_argument("--demo", action="store_true", help="使用演示模式（更快）")
    parser.add_argument("--no-dashboard", action="store_true", help="不启动仪表板")
    
    args = parser.parse_args()
    
    print_banner()
    print(f"📊 将生成 {args.size} 条英文护工服务记录")
    if args.demo:
        print("🏃 使用演示模式（快速生成）")
    print()
    
    # 确保在正确的目录
    project_root = Path(__file__).parent.parent
    dashboard_dir = Path(__file__).parent
    
    # 步骤1: 生成数据
    os.chdir(project_root)
    
    if args.demo:
        # 使用演示生成器
        cmd = [sys.executable, "dashboard/demo.py", "--records", str(args.size)]
        success, output = run_command(cmd, "生成演示数据", timeout=60)
    else:
        # 使用main_english.py
        cmd = [sys.executable, "main_english.py", "--size", str(args.size)]
        success, output = run_command(cmd, "生成高质量英文数据", timeout=180)
    
    if not success:
        print(f"\n💥 数据生成失败: {output}")
        print("\n💡 建议尝试:")
        print("   python dashboard/quick_generate_and_view.py --demo --size 10")
        return False
    
    # 步骤2: 更新仪表板数据库
    os.chdir(dashboard_dir)
    cmd = [sys.executable, "data_aggregator.py"]
    success, output = run_command(cmd, "更新仪表板数据库", timeout=60)
    
    if not success:
        print(f"\n💥 数据库更新失败: {output}")
        return False
    
    # 解析数据聚合结果
    if "Total Records:" in output:
        for line in output.split('\n'):
            if any(keyword in line for keyword in ['Total Records:', 'Overall Score:', 'Status:']):
                print(f"   📈 {line.strip()}")
    
    print("\n" + "=" * 60)
    print("🎉 数据生成和更新完成!")
    print()
    print("✅ 完成的任务:")
    print(f"   • 生成了 {args.size} 条英文护工服务记录")
    print("   • 更新了仪表板数据库")
    print("   • 准备好在仪表板中查看")
    
    if not args.no_dashboard:
        print("\n🚀 正在启动仪表板...")
        print("   URL: http://localhost:8501")
        print("   按 Ctrl+C 停止仪表板")
        print()
        
        # 给用户一点时间看到信息
        time.sleep(2)
        
        # 启动仪表板
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
        try:
            run_command(cmd, "启动仪表板", capture_output=False)
        except KeyboardInterrupt:
            print("\n✅ 仪表板已停止")
    else:
        print("\n🌐 要查看仪表板，请运行:")
        print("   python dashboard/start_simple.py")
        print("   然后打开: http://localhost:8501")
    
    print("\n🔄 要生成更多数据，请再次运行:")
    print(f"   python dashboard/quick_generate_and_view.py --size {args.size}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        sys.exit(1)
