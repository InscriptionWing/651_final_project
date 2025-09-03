@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo 🚀 一键生成英文护工数据并查看仪表板
echo    Generate English Carer Data ^& View Dashboard
echo ============================================================
echo.

cd /d "%~dp0.."

echo 🎯 选择生成模式:
echo    1. 快速演示模式 (15条记录, 约1分钟)
echo    2. 标准模式 (30条记录, 约3分钟) 
echo    3. 大批量模式 (100条记录, 约10分钟)
echo    4. 自定义数量
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    echo 🏃 启动快速演示模式...
    python dashboard\quick_generate_and_view.py --demo --size 15
) else if "%choice%"=="2" (
    echo ⚡ 启动标准模式...
    python dashboard\quick_generate_and_view.py --size 30
) else if "%choice%"=="3" (
    echo 🚀 启动大批量模式...
    python dashboard\quick_generate_and_view.py --size 100
) else if "%choice%"=="4" (
    set /p custom_size="请输入记录数量: "
    echo 🎯 启动自定义模式 (!custom_size! 条记录)...
    python dashboard\quick_generate_and_view.py --size !custom_size!
) else (
    echo ❌ 无效选择，使用默认标准模式
    python dashboard\quick_generate_and_view.py --size 30
)

echo.
echo 🎉 操作完成! 按任意键退出...
pause >nul
