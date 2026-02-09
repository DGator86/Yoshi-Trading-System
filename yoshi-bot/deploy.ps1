# Yoshi-Bot Optimized Deployment Script for Windows
# Run this on your local Windows PC.

# Load parameters from .env
$envFile = Get-Content ".env"
$VPS_IP = ($envFile | Select-String "VPS_IP=").Line.Split("=")[1].Trim()
$DEST_DIR = "gnosis_particle_bot"

if (-not $VPS_IP) {
    Write-Host "❌ Error: VPS_IP not found in .env" -ForegroundColor Red
    exit
}

# 1. Clean up local state that shouldn't be synced
Write-Host "🧹 Cleaning up local caches..." -ForegroundColor Cyan
Remove-Item -Path ".mypy_cache" -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "__pycache__" -Recurse -ErrorAction SilentlyContinue
# Note: We do NOT remove .venv locally as the user might be using it.
# Instead, we will sync individual folders to avoid copying .venv

Write-Host "📡 Syncing Yoshi-Bot to VPS ($VPS_IP)..." -ForegroundColor Cyan

# Use a temporary staging folder to avoid copying .venv
$staging = "deploy_staging"
if (Test-Path $staging) { Remove-Item -Path $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging -Force | Out-Null

# Robocopy is more robust for large trees and allows easy excludes
robocopy . $staging /E /XD .venv .git .mypy_cache __pycache__ $staging /R:1 /W:1 | Out-Null

# Sync from staging
scp -r "${staging}/*" "root@${VPS_IP}:/root/${DEST_DIR}"

# Cleanup staging
Remove-Item -Path $staging -Recurse -Force

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Sync failed." -ForegroundColor Red
    exit
}

Write-Host "⚙️ Starting services on VPS..." -ForegroundColor Cyan

# Build the command as a single line string including pkill for clean restart
$remoteCmd = "cd /root/${DEST_DIR}; pkill -f scripts/kalshi_scanner.py || true; pkill -f scripts/monitor_vps.py || true; chmod +x vps_setup.sh; ./vps_setup.sh; mkdir -p logs; . venv/bin/activate; nohup python3 scripts/kalshi_scanner.py --symbol BTCUSDT --loop --interval 300 --threshold 0.10 > logs/scanner.log 2>&1 & nohup python3 scripts/monitor_vps.py > logs/monitor.log 2>&1 &"

ssh "root@${VPS_IP}" $remoteCmd

Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "Check logs on VPS: tail -f ~/gnosis_particle_bot/logs/scanner.log" -ForegroundColor Yellow
