@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo Python not found on PATH. Install Python 3 and try again.
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\python.exe -m pip install -U pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
echo Done. In Cursor: Python -^> Select Interpreter -^> .venv
