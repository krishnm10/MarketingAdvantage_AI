@echo off
REM ═══════════════════════════════════════════════════════════
REM 🧪 Windows Batch Script - Deduplication Testing
REM ═══════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════
echo 🧪 MarketingAdvantage AI - Deduplication Testing Suite
echo ════════════════════════════════════════════════════════
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.7+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requests library is installed
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required library: requests
    pip install requests
    if errorlevel 1 (
        echo ❌ Failed to install requests library
        pause
        exit /b 1
    )
)

echo ✅ Required libraries installed
echo.

REM Run the Python test script
echo 🚀 Starting deduplication tests...
echo.

python run_dedup_tests.py

if errorlevel 1 (
    echo.
    echo ❌ Tests failed or were interrupted
    pause
    exit /b 1
)

echo.
echo 🎉 Testing complete!
echo.
echo 📊 Next steps:
echo    1. Check results in database
echo    2. Run: python verify_dedup_results.py
echo.
pause
