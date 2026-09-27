@echo off
rem Abre o Corrida IA. Na primeira vez, cria o ambiente Python e instala as dependencias.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Preparando o ambiente pela primeira vez...
    python -m venv .venv || goto erro
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto erro
)
".venv\Scripts\python.exe" main.py %*
exit /b

:erro
echo.
echo Nao foi possivel preparar o ambiente. Verifique se o Python 3.10+ esta instalado.
pause
