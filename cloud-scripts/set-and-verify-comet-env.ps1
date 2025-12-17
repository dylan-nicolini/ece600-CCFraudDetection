# -------------------------------------------------
# One-time Comet ML environment setup + verification
# -------------------------------------------------

Write-Host "Setting Comet ML environment variables..." -ForegroundColor Cyan

# Set environment variables (USER scope)
setx COMET_API_KEY "iU27xMQWN5Wi4rc3VLC8E34Az"
setx COMET_WORKSPACE "dylan-nicolini"
setx COMET_PROJECT_NAME "ece600-ccfraud"

Write-Host ""
Write-Host "Environment variables written to USER scope." -ForegroundColor Green
Write-Host "You MUST open a new PowerShell window to use them." -ForegroundColor Yellow
Write-Host ""

Write-Host "After reopening PowerShell, verify with:" -ForegroundColor Cyan
Write-Host "----------------------------------------"

Write-Host "COMET_API_KEY        =" -NoNewline
Write-Host " `$env:COMET_API_KEY"

Write-Host "COMET_WORKSPACE      =" -NoNewline
Write-Host " `$env:COMET_WORKSPACE"

Write-Host "COMET_PROJECT_NAME   =" -NoNewline
Write-Host " `$env:COMET_PROJECT_NAME"

Write-Host ""
Write-Host "Or run:" -ForegroundColor Cyan
Write-Host "  echo `$env:COMET_API_KEY"
Write-Host "  echo `$env:COMET_WORKSPACE"
Write-Host "  echo `$env:COMET_PROJECT_NAME"
