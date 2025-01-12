@echo off
echo SmartSigma Portfolio Optimizer Setup
echo ===================================

rem Check if virtual environment exists and create if it doesn't
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

rem Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:menu
echo.
echo [Virtual Environment: ACTIVATED]
echo Please select an option:
echo 1. Install dependencies
echo 2. Update dependencies
echo 3. Run SmartSigma app
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto update_deps
if "%choice%"=="3" goto run_app
if "%choice%"=="4" goto end
echo Invalid choice. Please try again.
goto menu

:install_deps
echo Installing dependencies...
pip install -r requirements.txt
echo Dependencies installed successfully!
goto menu

:update_deps
echo Updating dependencies...
pip install --upgrade -r requirements.txt
echo Dependencies updated successfully!
goto menu

:run_app
echo Starting SmartSigma app...
streamlit run app.py
goto menu

:end
echo Thank you for using SmartSigma Portfolio Optimizer!
pause
