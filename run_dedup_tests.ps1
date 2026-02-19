# ═══════════════════════════════════════════════════════════
# 🧪 PowerShell Script - Deduplication Testing
# ═══════════════════════════════════════════════════════════

Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host "🧪 MarketingAdvantage AI - Deduplication Testing Suite" -ForegroundColor Blue
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.7+" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if requests library is installed
try {
    python -c "import requests" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "requests not installed"
    }
    Write-Host "✅ Required libraries installed" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing required library: requests" -ForegroundColor Yellow
    pip install requests
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install requests library" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ requests library installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Starting deduplication tests..." -ForegroundColor Cyan
Write-Host ""

# Run the Python test script
python run_dedup_tests.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Tests failed or were interrupted" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🎉 Testing complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Check results in database"
Write-Host "   2. Run: python verify_dedup_results.py"
Write-Host ""
Read-Host "Press Enter to exit"
