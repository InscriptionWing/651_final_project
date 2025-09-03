#!/usr/bin/env python3
"""
简单启动脚本
直接启动仪表板，无额外检查
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主启动函数"""
    print("🚀 启动 NDIS 护工数据流水线仪表板")
    print("=" * 50)
    
    # 切换到仪表板目录
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    print(f"📁 工作目录: {dashboard_dir}")
    
    # 检查 streamlit_app.py 是否存在
    if not (dashboard_dir / "streamlit_app.py").exists():
        print("❌ 错误: streamlit_app.py 文件不存在")
        return False
    
    print("✅ 找到 streamlit_app.py 文件")
    
    # 启动命令
    print("\n🔧 启动 Streamlit...")
    print("   地址: http://localhost:8501")
    print("   按 Ctrl+C 停止仪表板")
    print()
    
    try:
        # 使用简单的启动命令
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
        
        print("⏳ 正在启动...")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n✅ 仪表板已停止")
    except FileNotFoundError:
        print("❌ 错误: 未找到 streamlit 命令")
        print("   请运行: pip install streamlit")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            input("按回车键退出...")
    except Exception as e:
        print(f"💥 意外错误: {e}")
        input("按回车键退出...")



