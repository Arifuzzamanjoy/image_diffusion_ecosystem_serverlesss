"""
Test script for Video LoRA Trainer UI
This script validates the basic functionality and configuration loading.
"""

import os
import sys
import yaml
import tempfile
import shutil
from pathlib import Path

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "ai-toolkit")

def test_config_loading():
    """Test that WAN 2.1 config files can be loaded correctly."""
    print("🧪 Testing WAN 2.1 configuration loading...")
    
    configs_to_test = [
        "config/examples/train_lora_wan21_1b_24gb.yaml",
        "config/examples/train_lora_wan21_14b_24gb.yaml"
    ]
    
    for config_path in configs_to_test:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_path} loaded successfully")
                print(f"   Model: {config['config']['process'][0]['model']['name_or_path']}")
                print(f"   Architecture: {config['config']['process'][0]['model'].get('arch', 'Not specified')}")
            except Exception as e:
                print(f"❌ Failed to load {config_path}: {e}")
        else:
            print(f"⚠️  {config_path} not found")

def test_imports():
    """Test that all required imports are available."""
    print("\n🧪 Testing required imports...")
    
    required_modules = [
        ("gradio", None),
        ("PIL", "Image"),
        ("torch", None),
        ("yaml", None),
        ("slugify", "slugify"),
        ("transformers", "AutoProcessor"),
        ("huggingface_hub", "whoami")
    ]
    
    failed_imports = []
    
    for module_name, import_as in required_modules:
        try:
            if import_as:
                module = __import__(module_name, fromlist=[import_as])
                getattr(module, import_as)
            else:
                __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
            print(f"❌ Failed to import {module_name}: {e}")
        except AttributeError as e:
            failed_imports.append((module_name, str(e)))
            print(f"❌ Failed to find {import_as} in {module_name}: {e}")
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} import(s) failed!")
        print("You may need to install missing packages:")
        print("  pip install gradio pillow torch pyyaml python-slugify transformers huggingface_hub")
    else:
        print(f"\n✅ All imports successful!")

def test_dataset_creation():
    """Test dataset creation functionality."""
    print("\n🧪 Testing dataset creation...")
    
    try:
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some dummy image files
            test_images = []
            for i in range(3):
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                # Create a simple text file to simulate image
                with open(img_path, 'w') as f:
                    f.write(f"dummy image {i}")
                test_images.append(img_path)
            
            # Test the create_dataset function
            from video_lora_train_ui import create_dataset
            
            # Simulate inputs: images list + captions
            inputs = [test_images] + [f"Caption for image {i}" for i in range(3)]
            
            dataset_folder = create_dataset(*inputs)
            
            if os.path.exists(dataset_folder):
                print(f"✅ Dataset created at: {dataset_folder}")
                
                # Check if metadata.jsonl exists
                metadata_path = os.path.join(dataset_folder, "metadata.jsonl")
                if os.path.exists(metadata_path):
                    print("✅ metadata.jsonl created successfully")
                    with open(metadata_path, 'r') as f:
                        content = f.read()
                        print(f"   Metadata content preview: {content[:100]}...")
                else:
                    print("❌ metadata.jsonl not found")
                
                # Cleanup
                shutil.rmtree(dataset_folder)
            else:
                print(f"❌ Dataset folder not created")
            
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")

def test_config_update():
    """Test configuration update functionality."""
    print("\n🧪 Testing configuration update...")
    
    try:
        from video_lora_train_ui import recursive_update
        
        # Test recursive update function
        base_config = {
            "model": {"name": "test"},
            "train": {"steps": 1000}
        }
        
        update_config = {
            "model": {"quantize": True},
            "train": {"steps": 2000}
        }
        
        result = recursive_update(base_config, update_config)
        
        expected_steps = 2000
        expected_quantize = True
        
        if (result["train"]["steps"] == expected_steps and 
            result["model"]["quantize"] == expected_quantize and
            result["model"]["name"] == "test"):
            print("✅ Configuration update works correctly")
        else:
            print(f"❌ Configuration update failed: {result}")
            
    except Exception as e:
        print(f"❌ Configuration update test failed: {e}")

def main():
    """Run all tests."""
    print("🎬 Video LoRA Trainer UI - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("video_lora_train_ui.py"):
        print("❌ Please run this test from the ai-toolkit directory")
        return
    
    test_imports()
    test_config_loading()
    test_dataset_creation()
    test_config_update()
    
    print("\n" + "=" * 50)
    print("🎬 Test suite completed!")
    print("\nIf all tests passed, you can now run:")
    print("  python video_lora_train_ui.py")
    print("  or")
    print("  ./run_video_lora_ui.bat (Windows)")
    print("  ./run_video_lora_ui.sh (Linux/Mac)")

if __name__ == "__main__":
    main()
