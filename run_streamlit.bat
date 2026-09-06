@echo off
title Handwritten Digit Recognizer - Streamlit Web App
echo ====================================================================
echo Launching Streamlit Web Application...
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

echo Starting Streamlit server on http://localhost:8501 ...
"%ENV_PYTHON%" -m streamlit run app.py

pause
