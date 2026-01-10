#!/bin/bash

# This script sets up the Python virtual environment and installs all required dependencies.
# It ensures that specific, compatible versions of key libraries are used.
#
# Usage:
# 1. Make the script executable: chmod +x install.sh
# 2. Run the script: ./install.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Create and Activate Virtual Environment ---
echo "🚀 Creating Python virtual environment in './venv'..."
python3 -m venv venv

echo " activating the virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment created and activated."
echo ""

# --- 2. Install PyTorch for CUDA 12.6 ---
echo "🚀 Installing PyTorch, TorchVision, and TorchAudio for CUDA 12.6..."
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
echo "✅ PyTorch installed."
echo ""

# --- 3. Install Base Project Requirements ---
echo "🚀 Installing base dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✅ Base dependencies installed."
echo ""

# --- 4. Install and Pin Key Libraries for Compatibility ---
echo "🚀 Installing specific versions of transformers, diffusers, peft, and timm..."
pip install --upgrade \
    transformers==4.45.2 \
    diffusers==0.35.1 \
    peft==0.17.0 \
    timm==1.0.17 \
    accelerate \
    huggingface_hub
echo "✅ Key libraries pinned to compatible versions."
echo ""

# --- Completion ---
echo "🎉 Setup complete! All dependencies are installed."
echo "To use the environment, run 'source venv/bin/activate' in your terminal."