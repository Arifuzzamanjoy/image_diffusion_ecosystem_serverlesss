#!/bin/bash

# Video LoRA Trainer Launcher Script
echo "🎬 Starting Video LoRA Trainer UI..."
echo "Powered by Ostris' AI Toolkit"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "video_lora_train_ui.py" ]; then
    echo "❌ video_lora_train_ui.py not found. Please run this script from the ai-toolkit directory."
    exit 1
fi

# Check if ai-toolkit is properly set up
if [ ! -d "toolkit" ]; then
    echo "❌ AI Toolkit not found. Please ensure you're in the correct directory."
    exit 1
fi

echo "✅ Starting Video LoRA Training UI..."
echo "📝 This will open in your web browser"
echo "🔗 Interface will be available at http://localhost:7860"
echo ""

# Set environment variables for better performance
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run the video LoRA trainer
python video_lora_train_ui.py

echo ""
echo "🎬 Video LoRA Trainer UI closed."
