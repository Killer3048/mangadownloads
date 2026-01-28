@echo off
setlocal ENABLEEXTENSIONS

set "ROOT_DIR=%~dp0"
:: Support UNC paths (e.g. \\wsl$\...) by mapping to a temp drive letter.
pushd "%ROOT_DIR%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Cannot change directory to: %ROOT_DIR%
  pause
  exit /b 1
)

if "%TIPER_PYTHON%"=="" (
  if exist "tiper\\.venv\\Scripts\\python.exe" (
    set "TIPER_PYTHON=tiper\\.venv\\Scripts\\python.exe"
  ) else (
    set "TIPER_PYTHON=python"
  )
)
if "%TIPER_HOST%"=="" set "TIPER_HOST=127.0.0.1"
if "%TIPER_PORT%"=="" set "TIPER_PORT=8765"

if "%START_OCR_WORKER%"=="" set "START_OCR_WORKER=1"
if "%OCR_WORKER_PYTHON%"=="" (
  if exist "tiper\\ocr_worker\\.venv\\Scripts\\python.exe" (
    set "OCR_WORKER_PYTHON=tiper\\ocr_worker\\.venv\\Scripts\\python.exe"
  ) else (
    set "OCR_WORKER_PYTHON=python"
  )
)
if "%OCR_WORKER_HOST%"=="" set "OCR_WORKER_HOST=127.0.0.1"
if "%OCR_WORKER_PORT%"=="" set "OCR_WORKER_PORT=8777"

if "%PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK%"=="" set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1"

if "%START_OCR_WORKER%"=="1" (
  if "%TIPER_OCR_WORKER_URL%"=="" set "TIPER_OCR_WORKER_URL=http://%OCR_WORKER_HOST%:%OCR_WORKER_PORT%"
)

echo [+] Root: %ROOT_DIR%
echo [+] TIPER_PYTHON: %TIPER_PYTHON%
echo [+] OCR_WORKER_PYTHON: %OCR_WORKER_PYTHON%

%TIPER_PYTHON% -c "import fastapi, uvicorn, pandas" >nul 2>nul
if errorlevel 1 (
  echo [!] Python deps missing for tiper server.
  echo     Run: %TIPER_PYTHON% -m pip install -r tiper\requirements.txt
  pause
  exit /b 1
)

if "%START_OCR_WORKER%"=="1" (
  %OCR_WORKER_PYTHON% -c "import fastapi, uvicorn, paddleocr, paddle" >nul 2>nul
  if errorlevel 1 (
    echo [!] Python deps missing for OCR worker.
    echo     Create a separate env and install: tiper\ocr_worker\requirements.txt ^(+ paddlepaddle/paddlepaddle-gpu^)
    pause
    exit /b 1
  )

  echo [+] Starting OCR worker: http://%OCR_WORKER_HOST%:%OCR_WORKER_PORT%
  start "TypeR OCR Worker" cmd /k "pushd \"%ROOT_DIR%\" && set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=%PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK%&& \"%OCR_WORKER_PYTHON%\" -m uvicorn app:app --app-dir tiper\\ocr_worker --host %OCR_WORKER_HOST% --port %OCR_WORKER_PORT%"
)

echo [+] Starting tiper server: http://%TIPER_HOST%:%TIPER_PORT%
start "TypeR tiper server" cmd /k "pushd \"%ROOT_DIR%\" && set TIPER_HOST=%TIPER_HOST%&& set TIPER_PORT=%TIPER_PORT%&& set TIPER_OCR_WORKER_URL=%TIPER_OCR_WORKER_URL%&& \"%TIPER_PYTHON%\" -m tiper.server"

echo [+] Done. Close the opened windows to stop the servers.
popd >nul 2>&1
endlocal
