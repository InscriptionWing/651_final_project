#!/usr/bin/env python3
"""
运行英文数据生成和仪表板集成脚本
专门配置为与 main_english.py 生成的数据完全兼容
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import time
import json

def print_header():
    """打印头部信息"""
    print("=" * 60)
    print("🇬🇧 NDIS English Data Generation & Dashboard Integration")
    print("   Compatible with main_english.py and pure_llm_english_generator.py")
    print("=" * 60)
    print()

def check_english_data_files():
    """检查英文数据相关文件是否存在"""
    project_root = Path(__file__).parent.parent
    required_files = [
        "main_english.py",
        "pure_llm_english_generator.py", 
        "english_data_schema.py"
    ]
    
    print("🔍 Checking English data generation files...")
    missing_files = []
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All English data generation files found")
    return True

def generate_english_data(size=20):
    """生成英文数据"""
    print(f"\n🤖 Generating {size} English carer service records...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        # 运行英文数据生成
        cmd = [sys.executable, "main_english.py", "--size", str(size)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ English data generation completed successfully!")
            
            # 解析输出以获取生成的文件信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Generated records:' in line or 'json:' in line or 'jsonl:' in line or 'csv:' in line:
                    print(f"   {line.strip()}")
            
            return True
        else:
            print("❌ English data generation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data generation timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running data generation: {e}")
        return False

def update_dashboard_data():
    """更新仪表板数据"""
    print("\n📊 Updating dashboard data...")
    
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    try:
        # 运行数据聚合
        cmd = [sys.executable, "data_aggregator.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dashboard data updated successfully!")
            
            # 解析输出查看聚合结果
            if "Data aggregation completed successfully" in result.stdout:
                print("   Data aggregation: ✅ Success")
            
            return True
        else:
            print("❌ Dashboard data update failed!")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error updating dashboard data: {e}")
        return False

def check_latest_data():
    """检查最新生成的数据文件"""
    print("\n📁 Checking latest generated data files...")
    
    output_dir = Path(__file__).parent.parent / "output"
    
    if not output_dir.exists():
        print("❌ Output directory not found")
        return False
    
    # 查找最新的英文数据文件
    english_files = list(output_dir.glob("*english*.json*"))
    pure_llm_files = list(output_dir.glob("*pure_llm*.json*"))
    
    all_files = english_files + pure_llm_files
    
    if not all_files:
        print("❌ No English data files found")
        return False
    
    # 按修改时间排序，获取最新的
    latest_files = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[:3]
    
    print("📊 Latest English data files:")
    for file_path in latest_files:
        size = file_path.stat().st_size
        mtime = time.ctime(file_path.stat().st_mtime)
        print(f"   📄 {file_path.name}")
        print(f"      Size: {size:,} bytes, Modified: {mtime}")
        
        # 如果是JSON文件，显示记录数
        if file_path.suffix == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"      Records: {len(data)}")
                    else:
                        print(f"      Records: 1")
            except:
                pass
    
    return True

def start_dashboard():
    """启动仪表板"""
    print("\n🚀 Starting dashboard...")
    
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    print("   Dashboard URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="English Data Generation & Dashboard Integration")
    parser.add_argument("--size", type=int, default=20, help="Number of records to generate")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation, only update dashboard")
    parser.add_argument("--check-only", action="store_true", help="Only check files and data, don't run anything")
    
    args = parser.parse_args()
    
    print_header()
    
    # 检查必需文件
    if not check_english_data_files():
        print("\n❌ Missing required files. Please ensure you have:")
        print("   - main_english.py")
        print("   - pure_llm_english_generator.py") 
        print("   - english_data_schema.py")
        return False
    
    # 检查现有数据
    check_latest_data()
    
    if args.check_only:
        print("\n✅ File and data check completed")
        return True
    
    # 生成数据（除非跳过）
    if not args.skip_generation:
        if not generate_english_data(args.size):
            print("\n❌ Data generation failed. Cannot proceed.")
            return False
    
    # 更新仪表板数据
    if not update_dashboard_data():
        print("\n❌ Dashboard data update failed.")
        return False
    
    # 显示成功信息
    print("\n" + "=" * 60)
    print("🎉 English Data Generation & Dashboard Integration Complete!")
    print()
    print("📊 What's available now:")
    print("   ✅ Fresh English carer service records")
    print("   ✅ Updated dashboard with latest data")
    print("   ✅ Real-time KPIs and quality metrics")
    print("   ✅ English narrative analysis")
    print()
    print("🚀 Ready to start dashboard!")
    
    # 询问是否启动仪表板
    try:
        start_dashboard_input = input("Start dashboard now? (Y/n): ").lower().strip()
        if start_dashboard_input in ['', 'y', 'yes']:
            start_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
