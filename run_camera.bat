@echo off
title CNN Handwritten Digit Recognizer - Camera Live Feed
echo ====================================================================
echo Starting Real-Time Handwritten Digit Recognizer (Camera Feed)...
echo ====================================================================

set ENV_PYTHON=C:\Users\LEGION\anaconda3\envs\tf_env\python.exe

if not exist "%ENV_PYTHON%" (
    echo [ERROR] Python environment not found at: %ENV_PYTHON%
    pause
    exit /b 1
)

if not exist "mnist_cnn_model.keras" (
    echo [INFO] Trained model not found. Training model now...
    "%ENV_PYTHON%" train_model.py
    if errorlevel 1 (
        echo [ERROR] Training failed.
        pause
        exit /b 1
    )
)

echo Launching Camera Application...
"%ENV_PYTHON%" realtime_detector.py %*

pause
