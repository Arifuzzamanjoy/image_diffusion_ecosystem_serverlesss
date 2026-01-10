#!/bin/bash
# 🌟 World-Class FLUX LoRA Trainer Launch Script

echo "🌟 Starting World-Class FLUX LoRA Trainer..."
echo "📍 Working Directory: $(pwd)"

# Activate virtual environment if it exists
if [ -f "../venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source ../venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if we're in the right directory
if [ ! -f "world_class_flux_trainer.py" ]; then
    echo "❌ Error: world_class_flux_trainer.py not found!"
    echo "📂 Please run this script from the ai-toolkit directory"
    exit 1
fi

# Launch the trainer
echo "🚀 Launching World-Class FLUX LoRA Trainer..."
echo ""
python world_class_flux_trainer.py

echo ""
echo "👋 World-Class FLUX LoRA Trainer session ended."