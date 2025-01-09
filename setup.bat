@echo off
echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Setup complete! Your virtual environment is now ready.
echo To start working, run: venv\Scripts\activate.bat

cmd /k
