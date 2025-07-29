@echo off
REM Overnight Prediction Batch File
REM Run this to start the overnight prediction system

echo Starting Overnight Prediction System...
echo.

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Run the overnight predictor
python overnight_predictor.py

echo.
echo Overnight prediction complete!
echo Check the output directory for results.
pause
