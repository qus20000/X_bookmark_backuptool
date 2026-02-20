$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$debugPort = 9222
$chromeCandidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
)
$chromeExe = $chromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $chromeExe) {
    Write-Host "[ERROR] chrome.exe not found."
    Write-Host "Please install Chrome or edit run_bookmark_tool.ps1 with your chrome.exe path."
    Read-Host "Press Enter to exit"
    exit 1
}

$userDataDir = "$env:LocalAppData\Google\Chrome\User Data"
$profileDir = "Default"

Write-Host "[INFO] Starting Chrome with remote debugging port $debugPort..."
Start-Process -FilePath $chromeExe -ArgumentList @(
    "--remote-debugging-port=$debugPort",
    "--user-data-dir=$userDataDir",
    "--profile-directory=$profileDir"
)

Write-Host "[INFO] Waiting for Chrome startup..."
Start-Sleep -Seconds 2

$pythonExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
Write-Host "[INFO] Running X_bookmark_backuptool.py..."
& $pythonExe "X_bookmark_backuptool.py"

Write-Host ""
Read-Host "[INFO] Finished. Press Enter to close"
