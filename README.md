# FLUX LoRA Trainer & Image Generator

This project provides tools for training FLUX LoRA models and generating images.

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Git
- Hugging Face account and token

## 🔑 Setup Hugging Face Token

1. Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **READ** permissions
3. Copy your token
4. Create a `.env` file from the example:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your token
nano .env
```
HF_TOKEN=your_huggingface_token_here

Replace `your_huggingface_token_here` with your actual Hugging Face token.

## 🚀 Installation & Setup

### Step 1: Make Shell Scripts Executable

First, make all shell scripts executable:

```bash
# Navigate to project directory
cd /root/Lora/Lora_Trainer_Imgen_Flux

# Make all shell scripts executable
chmod +x *.sh
```

### Step 2: Install AI Toolkit

Run the installation script to set up the virtual environment and dependencies:

```bash
./install_ai_toolkit.sh
```

This script will:
- Clone the ai-toolkit repository
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all required dependencies

## 🎯 Available Scripts

### 1. FLUX Training UI
Launch the Gradio-based training interface:

```bash
./run_flux_ui.sh
```

This will start a web interface for training FLUX LoRA models.

### 2. Automation Script
Run the automated image generation pipeline:

```bash
./run_automation12.sh
```

### 3. NSFW Image Generator
Launch the NSFW-capable image generator:

```bash
./run_nsfw.sh
```

This includes automatic model downloading from Google Drive.

## 📁 Project Structure

```
/root/Lora/Lora_Trainer_Imgen_Flux/
├── .env                      # Hugging Face token
├── README.md                 # This file
├── install_ai_toolkit.sh     # Installation script
├── run_flux_ui.sh           # Training UI launcher
├── run_automation12.sh      # Automation script runner
├── run_nsfw.sh              # NSFW generator runner
├── automation_12.py         # Automation pipeline
├── nsfw.py                  # NSFW image generator
└── ai-toolkit/              # AI toolkit repository
    ├── venv/                # Python virtual environment
    ├── flux_train_ui.py     # Training UI
    └── ...                  # Other toolkit files
```

## 🔧 Manual Commands

If you prefer to run commands manually:

### Activate Virtual Environment
```bash
cd /root/Lora/Lora_Trainer_Imgen_Flux/ai-toolkit
source venv/bin/activate
```

### Login to Hugging Face CLI
```bash
# Load token from .env and login
export $(grep -v '^#' ../.env | xargs)
echo "y" | huggingface-cli login --token "$HF_token"
```

### Run Python Scripts
```bash
# From project root, with venv activated
python automation_12.py
python nsfw.py
```

## 📊 Model Downloads

The NSFW generator will automatically download required models:
- **FLUX Transformer**: Downloaded from Google Drive (31FP16 version)
- **Text Encoders**: Downloaded from Hugging Face
- **VAE**: Downloaded from Hugging Face

Models are cached in `/home/caches` for optimal performance.

## 🔍 Troubleshooting

### Permission Issues
```bash
chmod +x *.sh
```

### Virtual Environment Issues
```bash
rm -rf ai-toolkit/venv
./install_ai_toolkit.sh
```

### CUDA Memory Issues
Restart your system or clear GPU memory:
```bash
nvidia-smi --gpu-reset
```

### Hugging Face Authentication
Make sure your token in `.env` has the correct permissions and format:
```
HF_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 📝 Notes

- The installation script only needs to be run once
- Each run script automatically activates the virtual environment
- Models are downloaded on first use and cached for subsequent runs
- All scripts include error checking and colored output for better UX

## 🚨 Important

- Keep your Hugging Face token secure and never commit it to version control
- Ensure you have sufficient disk space for model downloads (several GB)
- The NSFW generator requires CUDA for optimal performance

## 📞 Support

If you encounter issues:
1. Check that all `.sh` files are executable (`chmod +x *.sh`)
2. Verify your `.env` file contains a valid Hugging Face token
3. Ensure your system meets the prerequisites
4. Check GPU memory availability with `nvidia-smi`
