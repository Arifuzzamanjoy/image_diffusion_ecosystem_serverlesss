"""
FLUX LoRA Trainer Module
========================

Modularized version of the Advanced FLUX LoRA Trainer for the Unified AI Toolkit.
This module handles LoRA training with advanced dataset processing and GPU optimization.
"""

import os
import sys
import warnings
import tempfile
import zipfile
import json
import uuid
import shutil
import yaml
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import gradio as gr
from PIL import Image
from slugify import slugify

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

# Add ai-toolkit to path
current_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(current_dir))

# Import from ai-toolkit
from toolkit.job import get_job

MAX_IMAGES = 150

def recursive_update(d, u):
    """Recursively update dictionary"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def generate_dataset_preview(matched_data, max_preview=6):
    """Generate a visual preview of the dataset with image-caption matching"""
    
    if not matched_data:
        return "No dataset loaded yet."
    
    # Create a temporary directory for preview images
    preview_dir = tempfile.mkdtemp(prefix="dataset_preview_")
    preview_images = []
    
    try:
        for i, data in enumerate(matched_data[:max_preview]):
            image_path = data['image_path']
            caption = data['caption']
            
            # Copy image to preview directory with caption
            preview_image_path = os.path.join(preview_dir, f"preview_{i+1}.png")
            shutil.copy2(image_path, preview_image_path)
            preview_images.append((preview_image_path, f"Image {i+1}: {caption[:100]}..."))
        
        return preview_images
        
    except Exception as e:
        print(f"Error generating preview: {e}")
        return []

def validate_dataset_matching(matched_data):
    """Validate that images and captions are properly matched"""
    
    validation_report = {
        'total_images': len(matched_data),
        'valid_matches': 0,
        'empty_captions': 0,
        'sequential_order': True,
        'issues': []
    }
    
    expected_numbers = list(range(1, len(matched_data) + 1))
    actual_numbers = [data['image_number'] for data in matched_data]
    
    for i, data in enumerate(matched_data):
        # Check for empty captions
        if not data['caption'].strip():
            validation_report['empty_captions'] += 1
            validation_report['issues'].append(f"Empty caption for image {data['image_number']}")
        else:
            validation_report['valid_matches'] += 1
        
        # Check sequential numbering
        if data['image_number'] != expected_numbers[i]:
            validation_report['sequential_order'] = False
            validation_report['issues'].append(f"Non-sequential numbering: expected {expected_numbers[i]}, got {data['image_number']}")
    
    return validation_report

def create_dataset_preview(matched_data, matching_report):
    """Create visual preview of the first 6 image-caption matches for verification"""
    
    if not matched_data:
        return None, "No dataset loaded"
    
    # Create detailed matching report
    total_images = len(matched_data)
    avg_caption_length = sum(len(d['caption']) for d in matched_data) / total_images if total_images > 0 else 0
    
    report = f"""
## 📈 **Detailed Statistics:**
- ✅ **Successfully Matched**: {matching_report.get('successful_matches', 0)} pairs
- ❌ **Failed Matches**: {matching_report.get('failed_matches', 0)} items
- 📊 **Success Rate**: {(matching_report.get('successful_matches', 0) / matching_report.get('total_images_found', 1) * 100):.1f}%
- 📏 **Average Caption Length**: {avg_caption_length:.0f} characters

## 🔍 **First 10 Matched Pairs:**
"""
    
    # Add detailed verification for first 10 matches
    for i, data in enumerate(matched_data[:10]):
        status_icon = "✅" if data.get('status') == '✅ MATCHED' else "❌"
        report += f"""
**{i+1}. {data['image_name']}** {status_icon}
- 📷 **Image #{data.get('image_number', 'Unknown')}**
- 📝 **Caption**: *"{data['caption'][:120]}{'...' if len(data['caption']) > 120 else ''}"*
- 📐 **Length**: {len(data['caption'])} chars
"""
    
    if len(matched_data) > 10:
        report += f"\n... and **{len(matched_data) - 10}** more verified matches"
    
    # Add orphaned items if any
    if matching_report.get('orphaned_images'):
        report += f"""

## ⚠️ **Orphaned Images** ({len(matching_report['orphaned_images'])}):
"""
        for orphan in matching_report['orphaned_images'][:5]:
            report += f"- {orphan}\n"
        if len(matching_report['orphaned_images']) > 5:
            report += f"... and {len(matching_report['orphaned_images']) - 5} more\n"
    
    if matching_report.get('orphaned_captions'):
        report += f"""

## ⚠️ **Orphaned Captions** ({len(matching_report['orphaned_captions'])}):
"""
        for orphan in matching_report['orphaned_captions'][:5]:
            report += f"- {orphan}\n"
        if len(matching_report['orphaned_captions']) > 5:
            report += f"... and {len(matching_report['orphaned_captions']) - 5} more\n"
    
    # Create gallery preview (first 6 images with captions)
    gallery_items = []
    for i, data in enumerate(matched_data[:6]):
        try:
            gallery_items.append((data['image_path'], f"Image {i+1}: {data['caption'][:100]}"))
        except Exception as e:
            print(f"Error creating gallery item {i}: {e}")
    
    return gallery_items, report

def extract_captioning_data(images_zip_file, captions_json_file, progress=gr.Progress()):
    """Extract images from ZIP and captions from JSON, creating a unified dataset with detailed matching verification"""
    
    if images_zip_file is None or captions_json_file is None:
        raise gr.Error("Please upload both the images ZIP file and captions JSON file.")
    
    try:
        progress(0.1, desc="📦 Extracting images from ZIP...")
        
        # Create persistent temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="flux_trainer_")
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        with zipfile.ZipFile(images_zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        
        progress(0.3, desc="📋 Loading captions from JSON...")
        
        # Load captions JSON
        with open(captions_json_file.name, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        # Extract caption mappings
        if "captions" in captions_data:
            caption_mappings = captions_data["captions"]
        else:
            caption_mappings = captions_data
        
        progress(0.5, desc="🔍 Finding and matching images with captions...")
        
        # Find all image files in extracted directory (recursively)
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        # Sort image files by name for consistent ordering
        image_files.sort(key=lambda x: os.path.basename(x))
        
        progress(0.6, desc="🎯 Performing detailed matching verification...")
        
        # Create detailed matching data with verification
        matched_data = []
        matching_report = {
            "total_images_found": len(image_files),
            "total_captions_available": len(caption_mappings),
            "successful_matches": 0,
            "failed_matches": 0,
            "orphaned_images": [],
            "orphaned_captions": [],
            "matching_details": []
        }
        
        # Process each image file
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            image_number = None
            
            # Extract number from filename
            try:
                # Try different number extraction patterns
                import re
                number_match = re.search(r'(\d+)', image_name)
                if number_match:
                    image_number = int(number_match.group(1))
            except:
                pass
            
            # Look for matching caption
            caption = None
            caption_key = None
            
            # Try different matching strategies
            possible_keys = [
                image_name,
                os.path.splitext(image_name)[0],
                f"{image_number:04d}.png" if image_number else None,
                f"{image_number:04d}.jpg" if image_number else None,
                f"{image_number:04d}" if image_number else None,
            ]
            
            for key in possible_keys:
                if key and key in caption_mappings:
                    caption = caption_mappings[key]
                    caption_key = key
                    break
            
            if caption:
                matched_data.append({
                    "image_path": image_path,
                    "image_name": image_name,
                    "image_number": image_number or len(matched_data) + 1,
                    "caption": caption,
                    "caption_key": caption_key,
                    "status": "✅ MATCHED"
                })
                matching_report["successful_matches"] += 1
            else:
                matching_report["failed_matches"] += 1
                matching_report["orphaned_images"].append(image_name)
        
        # Check for captions without images
        used_caption_names = {os.path.basename(data["image_path"]) for data in matched_data}
        for caption_name in caption_mappings.keys():
            if caption_name not in used_caption_names:
                # Try to find if it matches any processed image
                found_match = False
                for data in matched_data:
                    if data["caption_key"] == caption_name:
                        found_match = True
                        break
                if not found_match:
                    matching_report["orphaned_captions"].append(caption_name)
        
        progress(0.8, desc="📊 Generating verification report...")
        
        # Sort matched data by image number for consistent ordering
        matched_data.sort(key=lambda x: x["image_number"])
        
        # Validate dataset
        if len(matched_data) < 2:
            raise gr.Error(f"Insufficient matched data: only {len(matched_data)} image-caption pairs found. Need at least 2.")
        elif len(matched_data) > MAX_IMAGES:
            raise gr.Error(f"Too many images: {len(matched_data)} found, maximum allowed is {MAX_IMAGES}")
        
        progress(0.9, desc="📋 Creating detailed verification report...")
        
        # Create comprehensive dataset info
        metadata = captions_data.get('metadata', {})
        
        dataset_info = f"""
✅ **Dataset Successfully Loaded and Verified!**

## 📊 **Matching Statistics:**
- 🖼️ **Images Found**: {matching_report['total_images_found']}
- 📝 **Captions Available**: {matching_report['total_captions_available']}
- ✅ **Successful Matches**: {matching_report['successful_matches']}
- ❌ **Failed Matches**: {matching_report['failed_matches']}
- 📈 **Match Success Rate**: {(matching_report['successful_matches']/matching_report['total_images_found']*100):.1f}%

## 📁 **Source Information:**
- 📂 **Original Folder**: {metadata.get('folder_name', 'Unknown')}
- 💬 **Captioning Prompt**: {metadata.get('prompt_used', 'Unknown')[:100]}{'...' if len(metadata.get('prompt_used', '')) > 100 else ''}
- ⏱️ **Processing Time**: {metadata.get('processing_time_seconds', 0):.1f} seconds

## 🎯 **Caption Quality:**
- 📏 **Average Caption Length**: {sum(len(d['caption']) for d in matched_data) / len(matched_data):.0f} characters
- 📊 **Caption Length Range**: {min(len(d['caption']) for d in matched_data)} - {max(len(d['caption']) for d in matched_data)} characters
        """
        
        progress(1.0, desc="🎯 Dataset ready for training!")
        
        # Create gallery preview
        gallery_items, detailed_report = create_dataset_preview(matched_data, matching_report)
        
        # Return matched data, detailed info, and enable training section
        return (
            matched_data,  # matched_data state
            dataset_info,  # dataset_info display
            detailed_report,  # detailed_report display
            gallery_items,  # gallery preview
            gr.update(visible=True),  # training_section visibility
            temp_dir  # temp_dir for cleanup
        )
            
    except Exception as e:
        error_msg = f"""
❌ **Error Processing Files**

**Error Details:** {str(e)}

**Common Solutions:**
1. Ensure ZIP file contains images with sequential names (0001.png, 0002.png, etc.)
2. Verify JSON file is from Advanced Image Captioning Pro
3. Check that image names in ZIP match caption keys in JSON
4. Ensure files are not corrupted
        """
        return None, error_msg, "", [], gr.update(visible=False), None

def create_training_dataset(matched_data, trigger_word="", trigger_position="beginning"):
    """Create dataset folder structure for training with optimized trigger positioning"""
    
    temp_dir_to_cleanup = None
    
    try:
        # Create unique dataset folder
        dataset_id = str(uuid.uuid4())
        destination_folder = f"datasets/{dataset_id}"
        os.makedirs(destination_folder, exist_ok=True)
        
        # Create metadata.jsonl file
        jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
        
        with open(jsonl_file_path, "w", encoding='utf-8') as jsonl_file:
            for i, data in enumerate(matched_data):
                # Copy image to destination
                source_path = data['image_path']
                file_name = f"{i+1:04d}.png"
                dest_path = os.path.join(destination_folder, file_name)
                
                # Convert and copy image
                with Image.open(source_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dest_path, "PNG", quality=95)
                
                # Process caption with trigger word
                caption = data['caption'].strip()
                
                if trigger_word and trigger_word.strip():
                    clean_trigger = trigger_word.strip()
                    
                    # Add trigger word based on position
                    if trigger_position == "beginning":
                        if not caption.lower().startswith(clean_trigger.lower()):
                            caption = f"{clean_trigger}, {caption}"
                    elif trigger_position == "end":
                        if not caption.lower().endswith(clean_trigger.lower()):
                            caption = f"{caption}, {clean_trigger}"
                    elif trigger_position == "random":
                        import random
                        words = caption.split()
                        if len(words) > 1:
                            insert_pos = random.randint(0, len(words))
                            words.insert(insert_pos, clean_trigger)
                            caption = " ".join(words)
                        else:
                            caption = f"{clean_trigger}, {caption}"
                
                # Clean up caption
                caption = caption.replace(', ,', ',').replace(',,', ',').replace('  ', ' ')
                
                # Create JSONL entry
                jsonl_entry = {
                    "file_name": file_name,
                    "prompt": caption
                }
                
                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
        
        return destination_folder
        
    except Exception as e:
        raise gr.Error(f"Error creating training dataset: {str(e)}")

def start_advanced_training(
    lora_name,
    concept_sentence,
    steps,
    lr,
    rank,
    linear_alpha,
    model_to_train,
    low_vram,
    batch_size,
    gradient_accumulation_steps,
    optimizer,
    save_dtype,
    train_dtype,
    guidance_scale,
    sample_steps,
    sample_every,
    caption_dropout_rate,
    resolution_512,
    resolution_768,
    resolution_1024,
    quantize,
    gradient_checkpointing,
    noise_scheduler,
    use_ema,
    ema_decay,
    matched_data,
    sample_1,
    sample_2,
    sample_3,
    trigger_position,
    use_more_advanced_options,
    more_advanced_options,
    progress=gr.Progress()
):
    """Start training with enhanced features and all advanced options"""
    
    if not lora_name:
        raise gr.Error("Please provide a LoRA name! This name must be unique.")
    
    if not matched_data:
        raise gr.Error("Please load dataset first by uploading images ZIP and captions JSON.")
    
    try:
        progress(0.05, desc="🔐 Checking Hugging Face authentication...")
        
        push_to_hub = True
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            hf_username = user_info.get('name', 'unknown')
            print(f"✅ Authenticated as: {hf_username}")
        except:
            push_to_hub = False
            print("⚠️ Hugging Face authentication not found. Models will be saved locally only.")
        
        progress(0.1, desc="📁 Creating training dataset...")
        
        # Create dataset folder with optimized trigger positioning
        dataset_folder = create_training_dataset(matched_data, concept_sentence, trigger_position)
        slugged_lora_name = slugify(lora_name)
        
        progress(0.2, desc="⚙️ Configuring training parameters...")
        
        # Load default config
        config_path_template = "config/examples/train_lora_flux_24gb.yaml"
        with open(config_path_template, "r") as f:
            config = yaml.safe_load(f)
        
        # Update config with user inputs
        config["config"]["name"] = slugged_lora_name
        config["config"]["process"][0]["model"]["low_vram"] = low_vram
        config["config"]["process"][0]["model"]["quantize"] = quantize
        config["config"]["process"][0]["train"]["skip_first_sample"] = True
        config["config"]["process"][0]["train"]["steps"] = int(steps)
        config["config"]["process"][0]["train"]["lr"] = float(lr)
        config["config"]["process"][0]["train"]["batch_size"] = int(batch_size)
        config["config"]["process"][0]["train"]["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
        config["config"]["process"][0]["train"]["optimizer"] = optimizer
        config["config"]["process"][0]["train"]["gradient_checkpointing"] = gradient_checkpointing
        config["config"]["process"][0]["train"]["noise_scheduler"] = noise_scheduler
        config["config"]["process"][0]["train"]["dtype"] = train_dtype
        
        # Network configuration
        config["config"]["process"][0]["network"]["linear"] = int(rank)
        config["config"]["process"][0]["network"]["linear_alpha"] = int(linear_alpha)
        
        # Dataset configuration
        config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
        config["config"]["process"][0]["datasets"][0]["caption_dropout_rate"] = float(caption_dropout_rate)
        
        # Resolution configuration
        resolutions = []
        if resolution_512:
            resolutions.append(512)
        if resolution_768:
            resolutions.append(768)
        if resolution_1024:
            resolutions.append(1024)
        if resolutions:
            config["config"]["process"][0]["datasets"][0]["resolution"] = resolutions
        
        # Save configuration
        config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
        config["config"]["process"][0]["save"]["dtype"] = save_dtype
        
        # Sampling configuration
        config["config"]["process"][0]["sample"]["guidance_scale"] = float(guidance_scale)
        config["config"]["process"][0]["sample"]["sample_steps"] = int(sample_steps)
        config["config"]["process"][0]["sample"]["sample_every"] = int(sample_every)
        
        # Add sample prompts if provided
        if any([sample_1, sample_2, sample_3]):
            config["config"]["process"][0]["sample"]["prompts"] = []
            if sample_1:
                config["config"]["process"][0]["sample"]["prompts"].append(sample_1)
            if sample_2:
                config["config"]["process"][0]["sample"]["prompts"].append(sample_2)
            if sample_3:
                config["config"]["process"][0]["sample"]["prompts"].append(sample_3)
        else:
            config["config"]["process"][0]["train"]["disable_sampling"] = True
        
        # Model selection
        if model_to_train == "schnell":
            config["config"]["process"][0]["model"]["name_or_path"] = "black-forest-labs/FLUX.1-schnell"
            config["config"]["process"][0]["model"]["assistant_lora_path"] = "ostris/FLUX.1-schnell-training-adapter"
            config["config"]["process"][0]["sample"]["sample_steps"] = 4
        
        progress(0.5, desc="🔧 Applying advanced configurations...")
        
        # Apply advanced options
        if use_more_advanced_options:
            try:
                more_advanced_options_dict = yaml.safe_load(more_advanced_options)
                config = recursive_update(config, more_advanced_options_dict)
                print("✅ Applied advanced configuration options")
            except Exception as e:
                print(f"⚠️ Warning: Could not parse advanced options: {e}")
        
        # EMA configuration
        if use_ema:
            config["config"]["process"][0]["train"]["ema_config"] = {
                "use_ema": True,
                "ema_decay": float(ema_decay)
            }
        
        progress(0.7, desc="💾 Saving configuration...")
        
        # Save config file
        config_filename = f"{slugged_lora_name}_config.yaml"
        config_path = f"tmp/{config_filename}"
        os.makedirs("tmp", exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        progress(0.8, desc="🚀 Starting training job...")
        
        # Start training
        job = get_job(config)
        job.run()
        
        progress(1.0, desc="✅ Training completed!")
        
        return f"""
🎉 **Training Started Successfully!**

**LoRA Name:** {lora_name}
**Config File:** {config_path}
**Dataset:** {len(matched_data)} images processed
**Output Location:** output/{slugged_lora_name}/

**Training Parameters:**
- Steps: {steps}
- Learning Rate: {lr}
- Rank: {rank}
- Batch Size: {batch_size}
- Model: {model_to_train}

The training is now running. Check the output folder for progress and results.
        """
        
    except Exception as e:
        return f"❌ **Training Failed:** {str(e)}"

def create_flux_trainer_interface() -> gr.Blocks:
    """Create the FLUX LoRA Trainer interface"""
    
    custom_css = """
    .trainer-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    
    .training-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .train-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
    }
    """
    
    with gr.Blocks(css=custom_css) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em;">
                🧠 Advanced FLUX LoRA Trainer
            </h1>
            <p style="font-size: 1.2em; color: #666;">
                Professional LoRA training with advanced dataset processing and GPU optimization
            </p>
        </div>
        """)
        
        # State variables
        matched_data = gr.State(None)
        temp_dir = gr.State(None)
        
        # Dataset Upload Section
        with gr.Row(elem_classes="trainer-container"):
            with gr.Column():
                gr.HTML('<h2 style="text-align: center;">📁 Dataset Upload & Processing</h2>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        images_zip = gr.File(
                            label="📦 Images ZIP File",
                            file_types=[".zip"],
                            file_count="single"
                        )
                    with gr.Column(scale=1):
                        captions_json = gr.File(
                            label="📋 Captions JSON File", 
                            file_types=[".json"],
                            file_count="single"
                        )
                
                process_btn = gr.Button("🔄 Process Dataset", variant="primary", size="lg")
                
                # Dataset info display
                dataset_info = gr.Markdown()
                dataset_gallery = gr.Gallery(
                    label="Dataset Preview",
                    show_label=True,
                    elem_id="dataset_gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
        
        # Training Configuration Section
        with gr.Row(visible=False, elem_classes="trainer-container") as training_section:
            with gr.Column():
                gr.HTML('<h2 style="text-align: center;">⚙️ Training Configuration</h2>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        lora_name = gr.Textbox(
                            label="LoRA Name",
                            placeholder="my-awesome-lora",
                            info="Unique name for your LoRA model"
                        )
                        concept_sentence = gr.Textbox(
                            label="Trigger Word/Concept",
                            placeholder="a person, anime style",
                            info="Main concept or trigger word for training"
                        )
                        
                    with gr.Column(scale=1):
                        steps = gr.Number(
                            label="Training Steps",
                            value=1000,
                            minimum=100,
                            maximum=10000,
                            step=100
                        )
                        lr = gr.Number(
                            label="Learning Rate",
                            value=1e-4,
                            minimum=1e-6,
                            maximum=1e-2,
                            step=1e-5
                        )
                
                with gr.Row():
                    with gr.Column():
                        rank = gr.Number(
                            label="Rank",
                            value=16,
                            minimum=4,
                            maximum=128,
                            step=4
                        )
                        linear_alpha = gr.Number(
                            label="Linear Alpha",
                            value=16,
                            minimum=1,
                            maximum=128,
                            step=1
                        )
                    with gr.Column():
                        batch_size = gr.Number(
                            label="Batch Size",
                            value=1,
                            minimum=1,
                            maximum=8,
                            step=1
                        )
                        model_to_train = gr.Radio(
                            choices=["dev", "schnell"],
                            value="dev",
                            label="Model to Train"
                        )
                
                # Advanced Options
                with gr.Accordion("🔧 Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            optimizer = gr.Dropdown(
                                choices=["adamw8bit", "adamw", "prodigy", "adafactor"],
                                value="adamw8bit",
                                label="Optimizer"
                            )
                            train_dtype = gr.Dropdown(
                                choices=["bf16", "fp16", "fp32"],
                                value="bf16",
                                label="Training Precision"
                            )
                            save_dtype = gr.Dropdown(
                                choices=["float16", "float32", "bf16"],
                                value="float16",
                                label="Save Precision"
                            )
                        with gr.Column():
                            low_vram = gr.Checkbox(
                                label="Low VRAM Mode",
                                value=True
                            )
                            quantize = gr.Checkbox(
                                label="Quantize Model",
                                value=True
                            )
                            gradient_checkpointing = gr.Checkbox(
                                label="Gradient Checkpointing",
                                value=True
                            )
                
                # Sample Prompts
                with gr.Accordion("🖼️ Sample Generation", open=False):
                    sample_1 = gr.Textbox(label="Sample Prompt 1", placeholder="a portrait of a person")
                    sample_2 = gr.Textbox(label="Sample Prompt 2", placeholder="a person in a landscape")
                    sample_3 = gr.Textbox(label="Sample Prompt 3", placeholder="a person, close-up")
                
                # Start Training
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", elem_classes="train-btn")
                training_status = gr.Markdown()
        
        # Event handlers
        process_btn.click(
            fn=extract_captioning_data,
            inputs=[images_zip, captions_json],
            outputs=[matched_data, dataset_info, gr.Markdown(), dataset_gallery, training_section, temp_dir]
        )
        
        train_btn.click(
            fn=start_advanced_training,
            inputs=[
                lora_name, concept_sentence, steps, lr, rank, linear_alpha,
                model_to_train, low_vram, batch_size, gr.Number(value=1),  # gradient_accumulation_steps
                optimizer, save_dtype, train_dtype,
                gr.Number(value=3.5),  # guidance_scale
                gr.Number(value=20),   # sample_steps  
                gr.Number(value=250),  # sample_every
                gr.Number(value=0.1),  # caption_dropout_rate
                gr.Checkbox(value=True),   # resolution_512
                gr.Checkbox(value=True),   # resolution_768
                gr.Checkbox(value=True),   # resolution_1024
                quantize, gradient_checkpointing,
                gr.Dropdown(value="flowmatch"),  # noise_scheduler
                gr.Checkbox(value=False),  # use_ema
                gr.Number(value=0.99),     # ema_decay
                matched_data, sample_1, sample_2, sample_3,
                gr.Dropdown(value="beginning"),  # trigger_position
                gr.Checkbox(value=False),        # use_more_advanced_options
                gr.Textbox(value="")             # more_advanced_options
            ],
            outputs=[training_status]
        )
    
    return interface