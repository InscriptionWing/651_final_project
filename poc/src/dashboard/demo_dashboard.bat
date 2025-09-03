@echo off
echo ========================================
echo NDIS Dashboard Demo Setup
echo ========================================
echo.

REM Change to dashboard directory
cd /d "%~dp0"

echo Step 1: Generating demo data...
python demo.py --records 150
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate demo data
    pause
    exit /b 1
)

echo.
echo Step 2: Starting dashboard...
echo The dashboard will open in your browser automatically.
echo Press Ctrl+C in this window to stop the dashboard.
echo.

python run_dashboard.py --mode dashboard

pause



