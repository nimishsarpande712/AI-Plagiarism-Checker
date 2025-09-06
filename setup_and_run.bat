@echo off
cls
echo ====================================
echo  Plagiarism Checker Setup
echo ====================================
echo.

:: Kill any existing Python processes
echo Stopping existing processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
timeout /t 2 /nobreak >nul

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    pause
    exit /b 1
)

:: Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting servers...
echo.

:: Start Flask server
echo Starting Flask backend server on port 5000...
start /B cmd /k "title Flask Backend && python app.py"

:: Wait for Flask server to start
timeout /t 3 /nobreak >nul

:: Start HTTP server for frontend
echo Starting frontend server on port 8000...
start cmd /k "title Frontend Server && cd Front && python -m http.server 8000"

echo.
echo ====================================
echo  Servers Started Successfully!
echo ====================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:8000
echo.
echo Press Ctrl+C in server windows to stop
echo.
pause
