@echo off
cls

:: Kill any existing Python processes
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

:: Wait for processes to close
timeout /t 2 /nobreak

:: Start Flask server in background
start /B pythonw app.py

:: Wait for Flask server to start
timeout /t 2 /nobreak

:: Start HTTP server for frontend
start cmd /k "cd Front && python -m http.server 8000"

echo.
echo ======================================
echo Flask API running on: http://localhost:5000
echo Frontend running on: http://localhost:8000
echo ======================================
echo.
echo Please use http://localhost:8000 to access the application
echo Press any key to exit...
pause >nul
echo Servers started! Access the application at http://localhost:8000
echo Press any key to exit...
pause >nul
