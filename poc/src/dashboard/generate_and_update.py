#!/usr/bin/env python3
"""
基于 main_english.py 的数据生成和仪表板更新脚本
一键完成数据生成、聚合和仪表板刷新
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import time
import json
from datetime import datetime

def print_header():
    """打印头部信息"""
    print("=" * 70)
    print("🔄 NDIS English Data Generation & Dashboard Update")
    print("   Generate → Aggregate → Display in Dashboard")
    print("=" * 70)
    print()

def generate_new_english_data(size=50, use_demo=False):
    """生成新的英文数据"""
    print(f"🤖 Step 1: Generating {size} new English carer service records...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        if use_demo:
            # 使用演示生成器（更快速）
            print("   Using demo generator for faster generation...")
            cmd = [sys.executable, "dashboard/demo.py", "--records", str(size)]
        else:
            # 使用main_english.py（高质量但可能较慢）
            print("   Using main_english.py for high-quality generation...")
            cmd = [sys.executable, "main_english.py", "--size", str(size)]
        
        print(f"   Running: {' '.join(cmd)}")
        
        # 设置较长的超时时间
        timeout = 60 if use_demo else 300  # 演示模式1分钟，标准模式5分钟
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print("✅ Data generation completed successfully!")
            
            # 解析输出查找生成的文件
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if any(keyword in line.lower() for keyword in ['generated', 'records:', 'json:', 'saved']):
                    print(f"   📊 {line.strip()}")
            
            return True, result.stdout
        else:
            print("❌ Data generation failed!")
            print("📝 STDOUT:", result.stdout[-500:])  # 最后500字符
            print("📝 STDERR:", result.stderr[-500:])
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"❌ Data generation timed out ({timeout//60} minutes)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error running data generation: {e}")
        return False, str(e)

def update_dashboard_database():
    """更新仪表板数据库"""
    print("\n📊 Step 2: Updating dashboard database...")
    
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    try:
        cmd = [sys.executable, "data_aggregator.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Dashboard database updated successfully!")
            
            # 解析输出获取聚合信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if any(keyword in line for keyword in ['Total Records:', 'Overall Score:', 'Status:']):
                    print(f"   📈 {line.strip()}")
            
            return True, result.stdout
        else:
            print("❌ Dashboard database update failed!")
            print("📝 STDERR:", result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ Database update timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        return False, str(e)

def get_latest_data_info():
    """获取最新数据信息"""
    print("\n📁 Step 3: Checking latest data...")
    
    output_dir = Path(__file__).parent.parent / "output"
    
    if not output_dir.exists():
        print("❌ Output directory not found")
        return None
    
    # 查找最新的数据文件
    all_files = list(output_dir.glob("*.json"))
    if not all_files:
        print("❌ No data files found")
        return None
    
    # 按修改时间排序，获取最新的
    latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        info = {
            "file_name": latest_file.name,
            "file_size": latest_file.stat().st_size,
            "modified_time": time.ctime(latest_file.stat().st_mtime),
            "record_count": len(data) if isinstance(data, list) else 1,
            "sample_record": data[0] if isinstance(data, list) and len(data) > 0 else data
        }
        
        print("✅ Latest data file information:")
        print(f"   📄 File: {info['file_name']}")
        print(f"   📊 Records: {info['record_count']}")
        print(f"   💾 Size: {info['file_size']:,} bytes")
        print(f"   🕒 Modified: {info['modified_time']}")
        
        if info['sample_record']:
            sample = info['sample_record']
            print(f"   👤 Sample Carer: {sample.get('carer_name', 'N/A')}")
            print(f"   🏥 Service Type: {sample.get('service_type', 'N/A')}")
            print(f"   ⏱️ Duration: {sample.get('duration_hours', 'N/A')} hours")
            print(f"   📍 Location: {sample.get('location_type', 'N/A')}")
        
        return info
        
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return None

def check_dashboard_status():
    """检查仪表板运行状态"""
    print("\n🖥️ Step 4: Checking dashboard status...")
    
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running at http://localhost:8501")
            return True
    except:
        pass
    
    print("⚠️ Dashboard not detected at http://localhost:8501")
    print("   You may need to start it manually with:")
    print("   python dashboard/start_simple.py")
    return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate English Data & Update Dashboard")
    parser.add_argument("--size", type=int, default=30, help="Number of records to generate")
    parser.add_argument("--demo", action="store_true", help="Use demo generator (faster)")
    parser.add_argument("--no-update", action="store_true", help="Generate only, don't update dashboard")
    parser.add_argument("--update-only", action="store_true", help="Update dashboard only, don't generate")
    
    args = parser.parse_args()
    
    print_header()
    
    generation_success = True
    update_success = True
    
    # 步骤1: 生成数据 (除非仅更新)
    if not args.update_only:
        generation_success, gen_output = generate_new_english_data(args.size, args.demo)
        
        if not generation_success:
            print(f"\n❌ Data generation failed. Output: {gen_output}")
            if not args.demo:
                print("\n💡 Tip: Try using --demo flag for faster generation:")
                print("   python generate_and_update.py --demo --size 20")
            return False
    else:
        print("⏭️ Skipping data generation (update-only mode)")
    
    # 步骤2: 更新仪表板 (除非禁用)
    if not args.no_update:
        update_success, update_output = update_dashboard_database()
        
        if not update_success:
            print(f"\n❌ Dashboard update failed. Output: {update_output}")
            return False
    else:
        print("⏭️ Skipping dashboard update (no-update mode)")
    
    # 步骤3: 检查数据信息
    get_latest_data_info()
    
    # 步骤4: 检查仪表板状态
    dashboard_running = check_dashboard_status()
    
    # 成功总结
    print("\n" + "=" * 70)
    print("🎉 Process Completed Successfully!")
    print()
    
    if generation_success and not args.update_only:
        print(f"✅ Generated {args.size} new English carer service records")
    
    if update_success and not args.no_update:
        print("✅ Updated dashboard database with new data")
    
    print("✅ Latest data information displayed")
    
    if dashboard_running:
        print("✅ Dashboard is running and ready to view")
        print("\n🌐 Access your updated dashboard at:")
        print("   👉 http://localhost:8501")
        print("\n📊 You should now see:")
        print("   • Updated KPI metrics")
        print("   • New English service records")
        print("   • Fresh data distributions")
        print("   • Real-time quality analysis")
    else:
        print("\n🚀 To view your data in the dashboard:")
        print("   1. Open a new terminal")
        print("   2. Run: python dashboard/start_simple.py")
        print("   3. Open browser: http://localhost:8501")
    
    print("\n🔄 To generate more data anytime:")
    print(f"   python dashboard/generate_and_update.py --size [NUMBER]")
    print(f"   python dashboard/generate_and_update.py --demo --size 20  # Faster")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
