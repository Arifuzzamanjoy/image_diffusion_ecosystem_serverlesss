@echo off
:: Video LoRA Trainer Launcher Script for Windows
title Video LoRA Trainer UI

echo 🎬 Starting Video LoRA Trainer UI...
echo Powered by Ostris' AI Toolkit
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Check if we're in the right directory
if not exist "video_lora_train_ui.py" (
    echo ❌ video_lora_train_ui.py not found. Please run this script from the ai-toolkit directory.
    pause
    exit /b 1
)

:: Check if ai-toolkit is properly set up
if not exist "toolkit" (
    echo ❌ AI Toolkit not found. Please ensure you're in the correct directory.
    pause
    exit /b 1
)

echo ✅ Starting Video LoRA Training UI...
echo 📝 This will open in your web browser
echo 🔗 Interface will be available at http://localhost:7860
echo.

:: Set environment variables for better performance
set HF_HUB_ENABLE_HF_TRANSFER=1

:: Run the video LoRA trainer
python video_lora_train_ui.py

echo.
echo 🎬 Video LoRA Trainer UI closed.
pause
