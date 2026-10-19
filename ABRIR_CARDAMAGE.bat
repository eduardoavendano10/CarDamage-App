@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title cardamage.ia - Presentacion para el profesor
color 0F

:MENU
cls
echo ========================================================
echo                 CARDAMAGE.IA
 echo                PRESENTACION FINAL
 echo ========================================================
echo.
echo   [1] Ver presentacion (resumen, datos, metodo, resultados)
echo   [2] Abrir aplicacion (demo en navegador)
echo   [3] Ver notebooks y evidencias
 echo   [4] Leer README completo
 echo   [5] Ver ficha del modelo
 echo   [6] Preparar dependencias de Python
 echo.
echo   [0] Salir
 echo.
choice /C 1234560 /N /M "Selecciona una opcion [1-6, 0]: "
if errorlevel 7 goto FIN
if errorlevel 6 goto INSTALAR
if errorlevel 5 goto MODELO
if errorlevel 4 goto README
if errorlevel 3 goto NOTEBOOKS
if errorlevel 2 goto APP
if errorlevel 1 goto PRESENTACION
goto MENU

:PRESENTACION
if not exist "PRESENTACION_PROFESOR.html" (
    echo ERROR: falta PRESENTACION_PROFESOR.html en la carpeta del proyecto.
    goto VOLVER
)
start "" "%CD%\PRESENTACION_PROFESOR.html"
goto MENU

:APP
cls
echo ==============================================
echo               INICIAR CARDAMAGE.IA
echo ==============================================
echo.
if not exist "app.py" (
    echo ERROR: app.py no esta en esta carpeta.
    goto VOLVER
)
call :ENCONTRAR_PYTHON
if errorlevel 1 goto VOLVER
"%PYTHON_EXE%" %PY_FLAGS% -c "import gradio,torch,torchvision; from google import genai" >nul 2>nul
if errorlevel 1 (
    echo ERROR: faltan librerias. Revisa requirements.txt.
    echo Puedes intentar la opcion [6] del menu.
    goto VOLVER
)
if not exist "model\model.pt" echo AVISO: falta model\model.pt; la inferencia real no estara disponible.
if not exist "model\classes.json" echo AVISO: falta model\classes.json; la inferencia real no estara disponible.
if not exist "clave_local.py" if not defined GEMINI_API_KEY echo AVISO: Gemini necesita una clave API propia; la clasificacion local si puede funcionar.
echo.
echo La app abrira el navegador; este proceso es su servidor.
echo Para volver al menu, detenla con Ctrl+C.
echo.
"%PYTHON_EXE%" %PY_FLAGS% app.py
echo.
echo Aplicacion cerrada.
goto VOLVER

:NOTEBOOKS
if exist "notebooks" (
    start "" explorer.exe "%CD%\notebooks"
) else (
    echo No existe la carpeta notebooks.
    goto VOLVER
)
goto MENU

:README
if exist "README.md" (
    start "" notepad.exe "%CD%\README.md"
) else (
    echo No se encontro README.md.
    goto VOLVER
)
goto MENU

:MODELO
if exist "model\model_info.json" (
    start "" notepad.exe "%CD%\model\model_info.json"
) else (
    echo No se encontro model\model_info.json.
    goto VOLVER
)
goto MENU

:INSTALAR
cls
echo ==============================================
echo              PREPARAR DEPENDENCIAS
echo ==============================================
echo.
echo Se requiere Python instalado y conexion a Internet.
if not exist "requirements.txt" (
    echo ERROR: no se encontro requirements.txt.
    goto VOLVER
)
if not exist ".venv\Scripts\python.exe" (
    call :ENCONTRAR_PYTHON
    if errorlevel 1 goto VOLVER
    echo Creando entorno virtual...
    "%PYTHON_EXE%" %PY_FLAGS% -m venv .venv
    if errorlevel 1 (
        echo No se pudo crear el entorno virtual.
        goto VOLVER
    )
)
echo.
echo Instalando librerias. Esta operacion puede tardar varios minutos.
"%CD%\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: instalacion incompleta. Revisa la salida de pip.
) else (
    echo Dependencias instaladas. Ya puedes usar la opcion [2].
)
goto VOLVER

:ENCONTRAR_PYTHON
set "PYTHON_EXE="
set "PY_FLAGS="
if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    exit /b 0
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PY_FLAGS=-3"
    exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    exit /b 0
)
echo ERROR: Python no esta disponible en este equipo.
exit /b 1

:VOLVER
echo.
pause
goto MENU

:FIN
endlocal
exit /b 0
