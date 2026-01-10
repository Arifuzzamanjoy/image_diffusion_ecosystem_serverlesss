#!/usr/bin/env python3
"""
🌟 FLUX LoRA Trainer

Professional LoRA training with optimized defaults and clean interface.

Features:
✨ Optimized templates for different training types
🖼️ Live training progress monitoring
🎯 Simple interface with advanced options available
📊 Real-time training analytics
🧠 Smart memory management
⚡ Fast training with quality results
💾 Professional export capabilities
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

# Performance optimizations
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Set up proper paths relative to this script
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir / "../ai-toolkit"))

def print_world_class_banner():
    """Print the trainer banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         🌟 FLUX LORA TRAINER 🌟                             ║
║                                                                              ║
║                  Professional LoRA Training Made Simple                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ ✨ Optimized Templates          🖼️ Live Progress Monitoring                 ║
║ 🎯 Simple & Clean Interface     📊 Training Analytics                       ║
║ 🧠 Smart Memory Management      ⚡ Fast & Quality Results                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check system requirements for world-class training"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for optimal performance")
        sys.exit(1)
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            
            if gpu_memory < 12:
                print("⚠️  Warning: <12GB VRAM detected. Low VRAM mode recommended.")
        else:
            print("⚠️  CUDA not available. CPU training will be very slow.")
    except ImportError:
        print("⚠️  PyTorch not found. Please install PyTorch with CUDA support.")
    
    print("✅ System check completed!")

def main():
    """Main entry point for the trainer"""
    print_world_class_banner()
    check_system_requirements()
    
    print("\n🚀 Initializing Components...")
    
    try:
        # Import the interface
        from modules.ui.world_class_simple_interface import WorldClassSimpleInterface
        print("✅ Modules loaded successfully")
        
        # Initialize the interface
        print("🎨 Building interface...")
        trainer_interface = WorldClassSimpleInterface()
        
        print("🌐 Launching trainer...")
        
        print("🎯 READY FOR LORA TRAINING!")
        print("📱 Interface will open in your browser")
        print("🔗 Share URL will be provided for remote access")
        print("=" * 80)
        
        # Launch
        trainer_interface.launch(
            server_name="0.0.0.0",
            share=True,
            show_error=True,
            quiet=True,
            inbrowser=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import components: {e}")
        print("💡 Install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("🔧 Check your configuration and try again")
        return False
    
    return True


def print_help():
    """Print help information"""
    help_text = """
🌟 FLUX LoRA Trainer - Help

USAGE:
    python world_class_flux_trainer.py [options]

FEATURES:
    ✨ Optimized Templates        - Pre-configured for different training types
    🖼️ Live Progress Gallery      - Real-time training visualization
    🎯 Clean Interface            - Simple by default, advanced when needed
    📊 Training Analytics         - Real-time metrics and monitoring
    🧠 Smart Memory Management    - Automatic VRAM optimization
    ⚡ Fast Training              - Optimized for speed and quality
    💾 Professional Export        - Multiple format support

TEMPLATES:
    👤 Character/Person           - Best for faces and people (1500 steps, Rank 32)
    🎨 Art Style                  - Optimized for artistic styles (1000 steps, Rank 16)
    🏃 Quick Test                 - Fast training for testing (500 steps)
    💎 Maximum Quality            - Best quality settings (3000 steps, Rank 64)

SYSTEM REQUIREMENTS:
    - Python 3.8+
    - CUDA-capable GPU (12GB+ VRAM recommended)
    - 16GB+ system RAM
    - Fast storage (SSD recommended)

For more information, visit the documentation.
    """
    print(help_text)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print_help()
            sys.exit(0)
        elif sys.argv[1] in ["-v", "--version", "version"]:
            print("🌟 FLUX LoRA Trainer v2.0")
            print("🚀 Professional LoRA training with optimized templates")
            sys.exit(0)
    
    # Launch the world-class trainer
    main()