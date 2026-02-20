@echo off
setlocal

REM Run from this script directory
cd /d "%~dp0"

set "DEBUG_PORT=9222"
set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe"

set "USER_DATA_DIR=%LocalAppData%\Google\Chrome\User Data"
set "PROFILE_DIR=Default"

if not exist "%CHROME_EXE%" (
  echo [ERROR] chrome.exe not found.
  echo Please install Chrome or edit run_bookmark_tool.bat with your chrome.exe path.
  pause
  exit /b 1
)

echo [INFO] Starting Chrome with remote debugging port %DEBUG_PORT%...
start "" "%CHROME_EXE%" --remote-debugging-port=%DEBUG_PORT% --user-data-dir="%USER_DATA_DIR%" --profile-directory="%PROFILE_DIR%"

echo [INFO] Waiting for Chrome startup...
timeout /t 2 /nobreak >nul

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [INFO] Running X_bookmark_backuptool.py...
"%PYTHON_EXE%" "X_bookmark_backuptool.py"

echo.
echo [INFO] Finished. Press any key to close.
pause >nul
endlocal
