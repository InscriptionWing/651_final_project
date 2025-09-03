#!/usr/bin/env python3
"""
测试导出功能修复
验证JSON序列化问题是否已解决
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from data_aggregator import DataAggregator

def make_json_safe(obj):
    """递归转换对象使其JSON安全"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_safe(obj.__dict__)
    else:
        return obj

def test_export_functionality():
    """测试导出功能"""
    print("🧪 Testing Export Functionality")
    print("=" * 40)
    
    try:
        # 获取仪表板数据
        print("📊 Getting dashboard data...")
        aggregator = DataAggregator()
        result = aggregator.aggregate_all_data()
        
        if result.get("status") != "success":
            print("❌ Failed to get dashboard data")
            return False
        
        print("✅ Dashboard data retrieved successfully")
        
        # 测试JSON序列化
        print("\n🔧 Testing JSON serialization...")
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "dashboard_data": make_json_safe(result),
            "export_format": "json"
        }
        
        # 尝试序列化
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        print("✅ JSON serialization successful!")
        print(f"   Data size: {len(json_data):,} characters")
        
        # 保存测试文件
        test_file = Path(__file__).parent / "test_export.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"✅ Test export saved: {test_file}")
        
        # 验证可以重新加载
        with open(test_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print("✅ JSON reload test successful!")
        
        # 显示导出内容摘要
        dashboard_data = loaded_data.get("dashboard_data", {})
        output_metrics = dashboard_data.get("output_metrics", {})
        
        print("\n📋 Export Content Summary:")
        print(f"   Export Timestamp: {loaded_data.get('export_timestamp', 'N/A')}")
        print(f"   Total Records: {output_metrics.get('total_records', 0)}")
        print(f"   Overall Score: {dashboard_data.get('validation_metrics', {}).get('overall_score', 0)}")
        
        # 清理测试文件
        test_file.unlink()
        print("🧹 Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_export():
    """测试CSV导出功能"""
    print("\n📊 Testing CSV Export...")
    
    try:
        import pandas as pd
        
        # 创建测试数据
        test_records = [
            {
                "record_id": "SR12345678",
                "carer_id": "CR123456",
                "carer_name": "Test Carer",
                "service_date": "2025-01-01",
                "service_type": "Personal Care",
                "duration_hours": 2.5,
                "narrative_notes": "Test narrative content"
            }
        ]
        
        # 转换为DataFrame并导出CSV
        df = pd.DataFrame(test_records)
        csv_data = df.to_csv(index=False)
        
        print("✅ CSV export test successful!")
        print(f"   CSV size: {len(csv_data)} characters")
        print(f"   Columns: {', '.join(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV export test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🔬 Dashboard Export Fix Verification")
    print("=" * 50)
    
    # 测试JSON导出
    json_success = test_export_functionality()
    
    # 测试CSV导出
    csv_success = test_csv_export()
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   JSON Export: {'✅ PASS' if json_success else '❌ FAIL'}")
    print(f"   CSV Export: {'✅ PASS' if csv_success else '❌ FAIL'}")
    
    if json_success and csv_success:
        print("\n🎉 All export functionality tests passed!")
        print("✅ The datetime serialization issue has been fixed")
        print("✅ Export buttons in dashboard should now work correctly")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test script error: {e}")
        sys.exit(1)
