#!/bin/bash
# filepath: /workspace/Lora_Trainer_Imgen_Flux/launch_professional_trainer.sh

# Professional FLUX LoRA Trainer Launch Script
# This script sets up the environment and launches the professional trainer

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                    🌟 PROFESSIONAL FLUX LORA TRAINER 🌟                    ║"
echo "║                                                                              ║"
echo "║               Professional-Grade LoRA Training Experience                   ║"
echo "║                                                                              ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                              ║"
echo "║ ✨ Research-Based Optimization    🖼️ Live Sample Gallery                    ║"
echo "║ 🎯 Professional Presets          📊 Advanced Analytics                      ║"
echo "║ 🧠 Smart Memory Management       ⚡ Adaptive Learning Rates                ║"
echo "║ 🔬 Parameter Validation          💾 Export & Sharing                       ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Set working directory
WORK_DIR="/workspace/Lora_Trainer_Imgen_Flux"
VENV_PATH="$WORK_DIR/venv"
AI_TOOLKIT_PATH="$WORK_DIR/ai-toolkit"

echo "🔍 Setting up Professional FLUX LoRA Trainer environment..."

# Change to working directory
cd "$WORK_DIR" || {
    echo "❌ Error: Could not change to directory $WORK_DIR"
    exit 1
}

echo "📂 Current directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "💡 Please create virtual environment first:"
    echo "   python -m venv venv"
    exit 1
fi

echo "🔧 Activating virtual environment..."
# Activate virtual environment
source "$VENV_PATH/bin/activate" || {
    echo "❌ Error: Could not activate virtual environment"
    exit 1
}

echo "✅ Virtual environment activated"

# Check if we're in the correct Python environment
if [[ "$VIRTUAL_ENV" != "$VENV_PATH" ]]; then
    echo "⚠️  Warning: Virtual environment may not be properly activated"
fi

echo "🔐 Setting up Hugging Face authentication..."

# Set Hugging Face token from environment variable or use placeholder
# Set your token: export HF_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. Please set it as an environment variable."
    echo "💡 Example: export HF_TOKEN='your_huggingface_token_here'"
fi

export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Silently authenticate with Hugging Face (suppress all warnings)
if [ -n "$HF_TOKEN" ]; then
    echo "🚀 Authenticating with Hugging Face..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" > /dev/null 2>&1 || true
fi

echo "✅ Authentication complete"

# Change to ai-toolkit directory
cd "$AI_TOOLKIT_PATH" || {
    echo "❌ Error: Could not change to ai-toolkit directory at $AI_TOOLKIT_PATH"
    exit 1
}

echo "📂 Changed to ai-toolkit directory: $(pwd)"

# Check if the main script exists
if [ ! -f "world_class_flux_trainer.py" ]; then
    echo "❌ Error: world_class_flux_trainer.py not found in current directory"
    echo "💡 Available Python files:"
    ls -la *.py 2>/dev/null || echo "   No Python files found"
    exit 1
fi

echo ""
echo "🚀 Launching Professional FLUX LoRA Trainer..."
echo "🎯 Features: Research-based defaults, EMA support, trigger word formatting"
echo "🌐 Interface will be available at: http://0.0.0.0:7860"
echo ""
echo "=================================================================================="
echo "🎯 READY FOR PROFESSIONAL LORA TRAINING!"
echo "📱 Interface will open in your browser automatically"
echo "🔗 Share URL will be provided for remote access"
echo "=================================================================================="
echo ""

# Run the Professional FLUX LoRA Trainer
python world_class_flux_trainer.py

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Professional FLUX LoRA Trainer completed successfully!"
else
    echo ""
    echo "❌ Professional FLUX LoRA Trainer exited with error code: $EXIT_CODE"
    echo "💡 Check the error messages above for troubleshooting"
fi

echo ""
echo "🌟 Thank you for using Professional FLUX LoRA Trainer!"
echo "🎯 For support and updates, visit: https://github.com/ostris/ai-toolkit"

exit $EXIT_CODE