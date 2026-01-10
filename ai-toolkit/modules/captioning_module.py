"""
Advanced Captioning Module
==========================

Modularized version of the Advanced Image Captioning Pro for the Unified AI Toolkit.
This module handles batch image captioning with GPU optimization and advanced AI models.
"""

import os
import sys
import gc
import io
import time
import json
import tempfile
import zipfile
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr
from PIL import Image, ImageOps, ImageEnhance
from PIL.ImageFile import LOAD_TRUNCATED_IMAGES
import torchvision.transforms.functional as TVF
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig
import psutil

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

# Enable loading of truncated images
LOAD_TRUNCATED_IMAGES = True

# Configuration
MODEL_PATH = "fancyfeast/llama-joycaption-alpha-two-vqa-test-1"
CACHE_DIR = "/root/.caches"
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'}

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    batch_size: int = 4
    max_workers: int = 8
    image_size: int = 384
    enable_amp: bool = True
    optimize_memory: bool = True
    use_flash_attention: bool = True
    prefetch_factor: int = 2
    compression_enabled: bool = False
    compression_quality: int = 85
    compression_optimize: bool = True

class AdvancedImageProcessor:
    """Advanced image processing with GPU optimization and compression"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def smart_compress_image(self, image: Image.Image, quality: int = 85, optimize: bool = True) -> Image.Image:
        """Intelligently compress image while maintaining quality"""
        if not self.config.compression_enabled:
            return image
        
        try:
            # Calculate original size
            original_size = len(image.tobytes())
            
            # For very small images, skip compression
            if original_size < 100000:
                return image
            
            # Save to bytes with compression
            buffer = io.BytesIO()
            
            # Choose format based on image characteristics
            if image.mode == 'RGBA' or 'transparency' in image.info:
                image.save(buffer, format='PNG', optimize=optimize)
            else:
                image.save(buffer, format='JPEG', quality=quality, optimize=optimize)
            
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            
            # Calculate compression ratio
            compressed_size = buffer.getbuffer().nbytes
            compression_ratio = compressed_size / original_size
            
            # Only use compressed version if it's significantly smaller
            if compression_ratio < 0.8:
                return compressed_image
            else:
                return image
                
        except Exception as e:
            print(f"⚠️ Compression failed: {e}")
            return image
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Optimized image preprocessing with optional compression"""
        try:
            # Load image with optimizations
            image = Image.open(image_path)
            
            # Handle transparency and convert to RGB
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply smart compression if enabled
            if self.config.compression_enabled:
                image = self.smart_compress_image(
                    image, 
                    self.config.compression_quality, 
                    self.config.compression_optimize
                )
            
            # High-quality resize with proper aspect ratio handling
            if image.size != (self.config.image_size, self.config.image_size):
                # Use high-quality resampling
                image = ImageOps.fit(
                    image, 
                    (self.config.image_size, self.config.image_size), 
                    Image.Resampling.LANCZOS
                )
            
            # Convert to tensor and normalize
            pixel_values = TVF.pil_to_tensor(image).float()
            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            
            return pixel_values
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {str(e)}")
            # Return a blank tensor as fallback
            return torch.zeros(3, self.config.image_size, self.config.image_size)

class GPUOptimizedCaptioner:
    """GPU-optimized captioning engine with advanced features"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.image_processor = AdvancedImageProcessor(config)
        self._model_loaded = False
        
    def _initialize_model(self):
        """Initialize model with advanced optimizations"""
        if self._model_loaded:
            return
            
        print("🚀 Initializing Advanced JoyCaption Model...")
        
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Advanced model configuration
        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
        }
        
        # Enable advanced optimizations if available
        if self.config.optimize_memory:
            model_kwargs["low_cpu_mem_usage"] = True
            
        try:
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, 
                use_fast=True, 
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
            
            # Load model
            print("🧠 Loading model...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_PATH, 
                **model_kwargs
            )
            
            # Enable optimizations
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                
            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print(f"⚠️ Model compilation failed: {e}")
            
            # Warm up the model
            self._warmup_model()
            
            self._model_loaded = True
            print("✅ Model initialization complete!")
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            raise e
        
    def _warmup_model(self):
        """Warm up the model with a dummy forward pass"""
        print("🔥 Warming up model...")
        try:
            # Create a dummy image tensor
            dummy_image = torch.randn(1, 3, self.config.image_size, self.config.image_size, 
                                    dtype=torch.bfloat16, device=self.device)
            dummy_text = "Describe this image."
            
            with torch.no_grad():
                # Prepare conversation
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a professional image captioning AI.",
                    },
                    {
                        "role": "user", 
                        "content": dummy_text,
                    },
                ]
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                
                # Add image
                inputs["pixel_values"] = dummy_image
                
                # Generate (short warmup)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            print("✅ Model warmup complete!")
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")
    
    def _prepare_conversation(self, prompt: str) -> List[Dict]:
        """Prepare conversation template"""
        return [
            {
                "role": "system",
                "content": "You are a professional image captioning AI that provides detailed, accurate, and engaging descriptions of images.",
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]
    
    @torch.no_grad()
    def _generate_caption_batch(self, pixel_values_batch: List[torch.Tensor], 
                               prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate captions for a batch of images"""
        if not pixel_values_batch:
            return []
            
        try:
            # Stack images into batch
            batch_pixel_values = torch.stack(pixel_values_batch).to(self.device)
            
            # Prepare conversations
            conversations = [self._prepare_conversation(prompt) for prompt in prompts]
            
            # Apply chat templates
            batch_prompts = [
                self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Add images
            inputs["pixel_values"] = batch_pixel_values
            
            # Generate captions
            with torch.cuda.amp.autocast(enabled=self.config.enable_amp):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 256),
                    temperature=generation_kwargs.get('temperature', 0.7),
                    top_p=generation_kwargs.get('top_p', 0.9),
                    top_k=generation_kwargs.get('top_k', 50),
                    repetition_penalty=generation_kwargs.get('repetition_penalty', 1.1),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode responses
            captions = []
            for i, output in enumerate(outputs):
                # Extract only the generated part
                input_length = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_length:]
                
                # Decode
                caption = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                caption = caption.strip()
                
                captions.append(caption)
            
            return captions
            
        except Exception as e:
            print(f"❌ Batch generation failed: {e}")
            return ["Error generating caption"] * len(pixel_values_batch)
    
    def generate_captions(self, image_paths: List[str], prompt: str, 
                         progress_callback=None, **generation_kwargs) -> Dict[str, str]:
        """Generate captions for multiple images with batching"""
        # Initialize model if not already loaded
        if not self._model_loaded:
            self._initialize_model()
            
        total_images = len(image_paths)
        captions = {}
        
        # Process images in batches
        for i in range(0, total_images, self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            if progress_callback:
                progress_callback((i + len(batch_paths)) / total_images, f"Processing batch {i//self.config.batch_size + 1}")
            
            # Preprocess batch images
            pixel_values_batch = []
            valid_paths = []
            
            for path in batch_paths:
                pixel_values = self.image_processor.preprocess_image(path)
                if pixel_values is not None:
                    pixel_values_batch.append(pixel_values)
                    valid_paths.append(path)
            
            if pixel_values_batch:
                # Generate captions for batch
                batch_prompts = [prompt] * len(pixel_values_batch)
                batch_captions = self._generate_caption_batch(pixel_values_batch, batch_prompts, **generation_kwargs)
                
                # Store results
                for path, caption in zip(valid_paths, batch_captions):
                    filename = os.path.basename(path)
                    captions[filename] = caption
            
            # Clear GPU cache periodically
            if i % (self.config.batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return captions

    def cleanup(self):
        """Cleanup model resources"""
        if self._model_loaded:
            try:
                del self.model
                del self.tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self._model_loaded = False
                print("🧹 Captioner model cleaned up")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

class SystemMonitor:
    """System resource monitoring"""
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                return {
                    "name": gpu_name,
                    "memory_total": f"{gpu_memory_total:.1f}GB",
                    "memory_used": f"{gpu_memory_used:.1f}GB",
                    "memory_percent": f"{gpu_memory_percent:.1f}%",
                    "load": "Available",
                    "temperature": "N/A"
                }
        except:
            pass
        return {"name": "Not available"}
    
    @staticmethod
    def get_cpu_info():
        """Get CPU information"""
        return {
            "cores": psutil.cpu_count(),
            "usage": f"{psutil.cpu_percent()}%",
            "memory": f"{psutil.virtual_memory().percent}%"
        }

def extract_and_process_images(zip_path: str, extract_to: str, compression_config: dict = None) -> Tuple[str, List[str]]:
    """Extract zip and process images with advanced handling and optional compression"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Find the main folder
    extracted_items = os.listdir(extract_to)
    main_folder = extract_to
    
    for item in extracted_items:
        item_path = os.path.join(extract_to, item)
        if os.path.isdir(item_path):
            main_folder = item_path
            break
    
    # Create processed folder
    processed_dir = os.path.join(main_folder, "processed_images")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find and convert images
    image_files = []
    for root, dirs, files in os.walk(main_folder):
        if "processed_images" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            if file_ext in SUPPORTED_IMAGE_FORMATS:
                image_files.append(file_path)
    
    # Sort for consistent ordering
    image_files.sort()
    
    converted_files = []
    total_original_size = 0
    total_compressed_size = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PNG with sequential naming
                output_filename = f"{i:04d}.png"
                output_path = os.path.join(processed_dir, output_filename)
                
                original_size = os.path.getsize(image_path)
                total_original_size += original_size
                
                # Apply compression if enabled
                if compression_config and compression_config.get('enabled', False):
                    # Save with compression
                    img.save(output_path, "PNG", optimize=compression_config.get('optimize', True))
                else:
                    # Save without compression
                    img.save(output_path, "PNG", quality=95)
                
                compressed_size = os.path.getsize(output_path)
                total_compressed_size += compressed_size
                
                converted_files.append(output_path)
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Print compression statistics
    if compression_config and compression_config.get('enabled', False) and total_original_size > 0:
        compression_ratio = total_compressed_size / total_original_size
        saved_space = total_original_size - total_compressed_size
        print(f"🗜️ Compression Results:")
        print(f"   Original: {total_original_size // 1024 // 1024}MB")
        print(f"   Compressed: {total_compressed_size // 1024 // 1024}MB")
        print(f"   Saved: {saved_space // 1024 // 1024}MB ({(1-compression_ratio)*100:.1f}%)")
    
    return processed_dir, converted_files

def create_processed_images_zip(processed_dir: str, folder_name: str) -> str:
    """Create a zip file containing all processed images"""
    try:
        # Create temporary zip file
        zip_filename = f"{folder_name}_processed_images.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all PNG files from processed directory
            for file in os.listdir(processed_dir):
                if file.endswith('.png'):
                    file_path = os.path.join(processed_dir, file)
                    zipf.write(file_path, file)
        
        return zip_path
    except Exception as e:
        print(f"Error creating processed images zip: {str(e)}")
        return None

def create_captioning_interface() -> gr.Blocks:
    """Create the Advanced Captioning interface"""
    
    # Configuration
    config = ProcessingConfig(
        batch_size=4,
        max_workers=8,
        image_size=384,
        enable_amp=True,
        optimize_memory=True,
        compression_enabled=False,
        compression_quality=85,
        compression_optimize=True
    )
    
    captioner = None
    monitor = SystemMonitor()
    
    def process_images_advanced(zip_file, prompt, temperature, top_p, top_k, 
                              repetition_penalty, max_new_tokens, batch_size,
                              enable_compression, compression_quality, compression_optimize,
                              progress=gr.Progress()):
        """Advanced image processing function with compression support"""
        
        nonlocal captioner
        
        if zip_file is None:
            return None, None, None, "❌ Please upload a ZIP file containing images", None
        
        if not prompt.strip():
            return None, None, None, "❌ Please provide a captioning prompt", None
        
        try:
            # Update configuration
            config.batch_size = batch_size
            config.compression_enabled = enable_compression
            config.compression_quality = compression_quality
            config.compression_optimize = compression_optimize
            
            # Initialize captioner if needed
            if captioner is None:
                captioner = GPUOptimizedCaptioner(config)
            
            start_time = time.time()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                progress(0.1, desc="📦 Extracting images from ZIP...")
                
                # Extract and process images
                compression_config = {
                    'enabled': enable_compression,
                    'quality': compression_quality,
                    'optimize': compression_optimize
                }
                processed_dir, image_files = extract_and_process_images(
                    zip_file.name, temp_dir, compression_config
                )
                
                if not image_files:
                    return None, None, None, "❌ No valid images found in ZIP file", None
                
                progress(0.3, desc=f"🔍 Processing {len(image_files)} images...")
                
                # Generate captions
                def progress_callback(ratio, desc):
                    progress(0.3 + ratio * 0.6, desc=desc)
                
                captions = captioner.generate_captions(
                    image_files, 
                    prompt,
                    progress_callback=progress_callback,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )
                
                progress(0.9, desc="📋 Creating output files...")
                
                # Create outputs
                folder_name = os.path.splitext(os.path.basename(zip_file.name))[0]
                
                # Create TXT report
                txt_content = f"Image Captioning Report\n"
                txt_content += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                txt_content += f"Prompt: {prompt}\n"
                txt_content += f"Total Images: {len(captions)}\n\n"
                
                for filename, caption in captions.items():
                    txt_content += f"{filename}: {caption}\n\n"
                
                txt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                txt_file.write(txt_content)
                txt_file.close()
                
                # Create JSON data
                json_data = {
                    "metadata": {
                        "folder_name": folder_name,
                        "total_images": len(captions),
                        "prompt_used": prompt,
                        "processing_time_seconds": time.time() - start_time,
                        "model_used": MODEL_PATH,
                        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    "captions": captions
                }
                
                json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
                json.dump(json_data, json_file, ensure_ascii=False, indent=2)
                json_file.close()
                
                # Create processed images ZIP
                images_zip_path = create_processed_images_zip(processed_dir, folder_name)
                
                progress(1.0, desc="✅ Processing complete!")
                
                # Create status report
                processing_time = time.time() - start_time
                status = f"""
✅ **Processing Complete!**

**📊 Summary:**
- **Images Processed**: {len(captions)}
- **Processing Time**: {processing_time:.1f} seconds
- **Average Time per Image**: {processing_time/len(captions):.2f} seconds
- **Model Used**: {MODEL_PATH.split('/')[-1]}

**📝 Prompt Used:**
"{prompt}"

**💾 Files Generated:**
- TXT Report: {len(captions)} captions
- JSON Data: Structured format with metadata
- Processed Images: Sequential PNG format
                """
                
                return txt_file.name, json_file.name, images_zip_path, status, get_system_status()
                
        except Exception as e:
            error_msg = f"❌ **Error during processing:**\n\n{str(e)}"
            return None, None, None, error_msg, get_system_status()
    
    def get_system_status():
        """Get current system status"""
        gpu_info = monitor.get_gpu_info()
        cpu_info = monitor.get_cpu_info()
        
        return f"""
        🖥️ **System Status:**
        
        **GPU:** {gpu_info.get('name', 'N/A')}
        - Memory: {gpu_info.get('memory_used', 'N/A')} / {gpu_info.get('memory_total', 'N/A')} ({gpu_info.get('memory_percent', 'N/A')})
        - Load: {gpu_info.get('load', 'N/A')}
        
        **CPU:** {cpu_info.get('cores', 'N/A')} cores
        - Usage: {cpu_info.get('usage', 'N/A')}
        - Memory: {cpu_info.get('memory', 'N/A')}
        """
    
    def cleanup_model():
        """Cleanup the captioner model"""
        nonlocal captioner
        if captioner:
            captioner.cleanup()
            captioner = None
        return "🧹 Model cleaned up successfully"
    
    # Custom CSS
    custom_css = """
    .captioning-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .settings-panel {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .process-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        color: white !important;
    }
    """
    
    with gr.Blocks(css=custom_css) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em;">
                📝 Advanced Image Captioning Pro
            </h1>
            <p style="font-size: 1.2em; color: #666;">
                Professional batch image captioning with GPU optimization and advanced AI models
            </p>
        </div>
        """)
        
        with gr.Row():
            # Input Section
            with gr.Column(scale=1, elem_classes="captioning-container"):
                gr.HTML('<h2 style="text-align: center;">📤 Input Configuration</h2>')
                
                zip_input = gr.File(
                    label="📦 Upload ZIP File",
                    file_types=[".zip"],
                    info="ZIP file containing images to caption"
                )
                
                prompt_input = gr.Textbox(
                    label="💬 Captioning Prompt",
                    placeholder="Describe this image in detail",
                    lines=3,
                    info="The prompt that will guide the captioning process"
                )
                
                # Advanced Settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                                label="Temperature", info="Controls creativity"
                            )
                            top_p = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                                label="Top P", info="Nucleus sampling"
                            )
                        with gr.Column():
                            top_k = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1,
                                label="Top K", info="Top-k sampling"
                            )
                            repetition_penalty = gr.Slider(
                                minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                                label="Repetition Penalty"
                            )
                    
                    with gr.Row():
                        max_new_tokens = gr.Number(
                            value=256, minimum=50, maximum=512, step=16,
                            label="Max New Tokens", info="Maximum caption length"
                        )
                        batch_size = gr.Number(
                            value=4, minimum=1, maximum=16, step=1,
                            label="Batch Size", info="Images processed simultaneously"
                        )
                
                # Compression Settings
                with gr.Accordion("🗜️ Compression Settings", open=False):
                    enable_compression = gr.Checkbox(
                        label="Enable Image Compression",
                        value=False,
                        info="Reduce file sizes while maintaining quality"
                    )
                    with gr.Row():
                        compression_quality = gr.Slider(
                            minimum=50, maximum=95, value=85, step=5,
                            label="Compression Quality"
                        )
                        compression_optimize = gr.Checkbox(
                            label="Optimize Compression",
                            value=True
                        )
                
                process_btn = gr.Button("🚀 Process Images", variant="primary", size="lg", elem_classes="process-btn")
                cleanup_btn = gr.Button("🧹 Cleanup Model", variant="secondary")
            
            # Output Section
            with gr.Column(scale=1, elem_classes="captioning-container"):
                gr.HTML('<h2 style="text-align: center;">📤 Results & Downloads</h2>')
                
                processing_status = gr.Markdown()
                
                with gr.Row():
                    txt_download = gr.File(label="📄 TXT Report", interactive=False)
                    json_download = gr.File(label="📋 JSON Data", interactive=False)
                
                images_zip_download = gr.File(label="🖼️ Processed Images ZIP", interactive=False)
                
                # System Status
                with gr.Accordion("📊 System Status", open=False):
                    system_status = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh Status")
        
        # Event handlers
        process_btn.click(
            fn=process_images_advanced,
            inputs=[
                zip_input, prompt_input, temperature, top_p, top_k,
                repetition_penalty, max_new_tokens, batch_size,
                enable_compression, compression_quality, compression_optimize
            ],
            outputs=[txt_download, json_download, images_zip_download, processing_status, system_status]
        )
        
        cleanup_btn.click(
            fn=cleanup_model,
            outputs=[processing_status]
        )
        
        refresh_btn.click(
            fn=get_system_status,
            outputs=[system_status]
        )
        
        # Auto-refresh system status on load
        interface.load(get_system_status, outputs=[system_status])
    
    return interface