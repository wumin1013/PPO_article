@echo off
setlocal enabledelayedexpansion

REM Portable Python resolver: prefer real interpreters, avoid WindowsApps stubs.
set "PYTHON_CANDIDATE="
set "SCRIPT_PATH=%~f0"

REM 1) search PATH for python.exe excluding WindowsApps stub and self
for /f "usebackq tokens=*" %%P in (`where python 2^>nul`) do (
    if /i "%%~xP"==".exe" (
        if /i not "%%~fP"=="%SCRIPT_PATH%" (
            echo %%P | findstr /i "WindowsApps" >nul
            if errorlevel 1 (
                set "PYTHON_CANDIDATE=%%P"
                goto :found
            )
        )
    )
)

REM 2) fallback to py launcher if available
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CANDIDATE=py"
    goto :found
)

REM 3) search common Anaconda/Miniconda/Python install locations
if not defined PYTHON_CANDIDATE if exist "%USERPROFILE%\anaconda3\python.exe" set "PYTHON_CANDIDATE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYTHON_CANDIDATE if exist "%USERPROFILE%\miniconda3\python.exe" set "PYTHON_CANDIDATE=%USERPROFILE%\miniconda3\python.exe"
if not defined PYTHON_CANDIDATE if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_CANDIDATE=%ProgramFiles%\Python311\python.exe"
if not defined PYTHON_CANDIDATE if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON_CANDIDATE=%ProgramFiles%\Python312\python.exe"
if not defined PYTHON_CANDIDATE if exist "%ProgramFiles(x86)%\Python311\python.exe" set "PYTHON_CANDIDATE=%ProgramFiles(x86)%\Python311\python.exe"
if not defined PYTHON_CANDIDATE if exist "%ProgramFiles(x86)%\Python312\python.exe" set "PYTHON_CANDIDATE=%ProgramFiles(x86)%\Python312\python.exe"

:found
if not defined PYTHON_CANDIDATE (
    echo [python.cmd] No usable Python found. Please install Python or add it to PATH.
    exit /b 9009
)

REM If using py launcher, let it handle the call; otherwise call resolved path
if /i "%PYTHON_CANDIDATE%"=="py" (
    py %*
    set "EXITCODE=!ERRORLEVEL!"
) else (
    "%PYTHON_CANDIDATE%" %*
    set "EXITCODE=!ERRORLEVEL!"
)

exit /b !EXITCODE!
