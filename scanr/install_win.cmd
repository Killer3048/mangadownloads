@echo off
setlocal ENABLEEXTENSIONS
:: Support UNC paths (e.g. \\wsl$\...) by mapping to a temp drive letter.
pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Cannot change directory to: %~dp0
  exit /b 1
)

:: Lance le script PowerShell en contournant la politique de sécurité (ExecutionPolicy)
:: et en passant le contrôle à PowerShell
PowerShell -NoProfile -ExecutionPolicy Bypass -File "install.ps1"
popd >nul 2>&1
endlocal
