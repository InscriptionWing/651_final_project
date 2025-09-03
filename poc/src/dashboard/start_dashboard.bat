@echo off
echo ========================================
echo NDIS Carer Data Pipeline Dashboard
echo ========================================
echo.

REM Change to dashboard directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if virtual environment exists
if exist "dashboard_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call dashboard_env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
    echo Consider creating a virtual environment with:
    echo python -m venv dashboard_env
    echo.
)

REM Install dependencies if needed
echo Checking dependencies...
python -c "import streamlit, plotly, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK
echo.

REM Run dashboard launcher
echo Starting dashboard...
python run_dashboard.py --mode dashboard

echo.
echo Dashboard stopped.
pause



