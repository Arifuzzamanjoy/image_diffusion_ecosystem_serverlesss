#!/usr/bin/env python3
"""
AI Toolkit Launcher
===================

Universal launcher for the AI Toolkit with multiple launch modes:
1. Unified Interface (all modules in one tabbed interface)
2. Individual Module Launchers (each module separately)
3. Help and Information

This launcher provides GPU-efficient options and flexible deployment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                            🚀 AI Toolkit Launcher                           ║
║                                                                              ║
║  Advanced FLUX Training • Image Captioning • AI Generation                  ║
║  GPU-Optimized • Modular Architecture • Professional Results                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_venv():
    """Check if virtual environment is active"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✅ Virtual environment active: {sys.prefix}")
        return True
    else:
        print("⚠️ Warning: No virtual environment detected")
        print("   It's recommended to use a virtual environment")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        ("gradio", "gr"),
        ("torch", "torch"),
        ("PIL", "PIL"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers")
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name} (missing)")
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def launch_unified():
    """Launch unified interface"""
    print("\n🚀 Launching Unified AI Toolkit Interface...")
    print("   All modules available in tabbed interface")
    print("   GPU resources managed automatically")
    
    try:
        subprocess.run([sys.executable, "unified_ai_toolkit.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch unified interface: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0

def launch_flux_trainer():
    """Launch FLUX LoRA Trainer"""
    print("\n🧠 Launching FLUX LoRA Trainer (Standalone)...")
    print("   Advanced LoRA training with dataset processing")
    
    try:
        subprocess.run([sys.executable, "launch_flux_trainer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch FLUX trainer: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0

def launch_captioning():
    """Launch Advanced Captioning"""
    print("\n📝 Launching Advanced Image Captioning Pro (Standalone)...")
    print("   Professional batch image captioning")
    
    try:
        subprocess.run([sys.executable, "launch_captioning.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch captioning: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0

def launch_generator():
    """Launch Image Generator"""
    print("\n🎨 Launching FLUX Image Generator (Standalone)...")
    print("   AI image generation with LoRA support")
    
    try:
        subprocess.run([sys.executable, "launch_generator.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch generator: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0

def show_interactive_menu():
    """Show interactive menu for launch options"""
    while True:
        print("\n" + "="*80)
        print("🚀 AI TOOLKIT LAUNCHER")
        print("="*80)
        print("Choose your launch mode:")
        print()
        print("1. 🌐 Unified Interface (All modules in tabs)")
        print("   • GPU-efficient resource management")
        print("   • Switch between modules seamlessly") 
        print("   • Shared authentication")
        print()
        print("2. 🧠 FLUX LoRA Trainer (Standalone)")
        print("   • Advanced dataset processing")
        print("   • Professional LoRA training")
        print("   • Full GPU utilization")
        print()
        print("3. 📝 Advanced Captioning (Standalone)")
        print("   • Batch image processing")
        print("   • Advanced AI models")
        print("   • Professional outputs")
        print()
        print("4. 🎨 FLUX Image Generator (Standalone)")
        print("   • High-quality image generation")
        print("   • LoRA model support")
        print("   • Batch processing")
        print()
        print("5. ℹ️  System Information")
        print("6. ❌ Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                return launch_unified()
            elif choice == "2":
                return launch_flux_trainer()
            elif choice == "3":
                return launch_captioning()
            elif choice == "4":
                return launch_generator()
            elif choice == "5":
                show_system_info()
            elif choice == "6":
                print("👋 Goodbye!")
                return 0
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except EOFError:
            print("\n👋 Goodbye!")
            return 0

def show_system_info():
    """Show system information"""
    print("\n" + "="*60)
    print("📊 SYSTEM INFORMATION")
    print("="*60)
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    
    # Virtual environment
    venv_active = check_venv()
    
    # Working directory
    current_dir = Path(__file__).parent.absolute()
    print(f"📂 Working Directory: {current_dir}")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps_ok = check_dependencies()
    
    # GPU info
    print("\n🖥️ GPU Information:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ Memory: {gpu_memory:.1f}GB")
        else:
            print("   ❌ CUDA not available")
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    # Hugging Face token
    print("\n🔐 Authentication:")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("   ✅ HF_TOKEN found in environment")
    else:
        print("   ⚠️ HF_TOKEN not found - may need manual authentication")
    
    print("\n" + "="*60)
    input("Press Enter to continue...")

def show_help():
    """Show help information"""
    help_text = """
🚀 AI Toolkit Launcher Help
===========================

LAUNCH MODES:
-------------

1. Unified Interface (--unified)
   • All modules in one tabbed interface
   • GPU resources managed automatically
   • Memory efficient - loads only active tab
   • Best for exploration and switching between tasks

2. Individual Modules (--flux-trainer, --captioning, --generator)  
   • Each module runs standalone
   • Full GPU utilization for single task
   • Best for focused, intensive work
   • Can run multiple instances if GPU memory allows

COMMAND LINE OPTIONS:
--------------------

python launcher.py [options]

Options:
  --unified              Launch unified interface
  --flux-trainer        Launch FLUX LoRA Trainer
  --captioning          Launch Advanced Captioning
  --generator           Launch Image Generator
  --info                Show system information
  --help                Show this help
  
If no options provided, interactive menu is shown.

EXAMPLES:
---------

# Interactive menu (recommended for beginners)
python launcher.py

# Direct launch modes
python launcher.py --unified
python launcher.py --flux-trainer
python launcher.py --captioning
python launcher.py --generator

# System check
python launcher.py --info

REQUIREMENTS:
------------

• Python 3.8+
• Virtual environment (recommended)
• CUDA-capable GPU (for best performance)
• Hugging Face token (for model downloads)

For detailed setup instructions, see README.md
    """
    print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Toolkit Launcher - Advanced FLUX Training, Captioning & Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--unified", action="store_true", 
                       help="Launch unified interface")
    parser.add_argument("--flux-trainer", action="store_true",
                       help="Launch FLUX LoRA Trainer")
    parser.add_argument("--captioning", action="store_true",
                       help="Launch Advanced Captioning")
    parser.add_argument("--generator", action="store_true", 
                       help="Launch Image Generator")
    parser.add_argument("--info", action="store_true",
                       help="Show system information")
    parser.add_argument("--help-detailed", action="store_true",
                       help="Show detailed help")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print_banner()
    
    # Handle command line arguments
    if args.help_detailed:
        show_help()
        return 0
    elif args.info:
        show_system_info()
        return 0
    elif args.unified:
        return launch_unified()
    elif args.flux_trainer:
        return launch_flux_trainer()
    elif args.captioning:
        return launch_captioning()
    elif args.generator:
        return launch_generator()
    else:
        # No arguments provided, show interactive menu
        return show_interactive_menu()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)