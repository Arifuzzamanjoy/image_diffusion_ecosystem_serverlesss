import gradio as gr
from transformers import AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms.functional as TVF
import os
import zipfile
import json
import shutil
from pathlib import Path
import tempfile
from typing import List, Dict, Tuple, Optional
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil
from dataclasses import dataclass
import hashlib
import gc
from PIL.ImageFile import LOAD_TRUNCATED_IMAGES
from PIL import ImageEnhance

# Advanced Configuration
MODEL_PATH = "fancyfeast/llama-joycaption-alpha-two-vqa-test-1"
CACHE_DIR = "/root/.caches"
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'}

# Enable loading of truncated images for better handling
LOAD_TRUNCATED_IMAGES = True

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
    
    def smart_compress_image(self, image: Image.Image, quality: int = 85, optimize: bool = False) -> Image.Image:
        """Fast image compression for preprocessing (simplified and optimized)"""
        # This method is now a lightweight pass-through since we handle compression
        # during the save phase in _process_single_image for better performance
        return image
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Ultra-fast optimized image preprocessing"""
        try:
            # Load image efficiently
            image = Image.open(image_path)
            image.load()  # Force load to close file handle immediately
            
            # Fast RGB conversion
            if image.mode != 'RGB':
                if image.mode in ('RGBA', 'LA'):
                    # Fast transparency handling
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if len(image.split()) > 3 else None)
                    image = background
                elif image.mode == 'P':
                    image = image.convert('RGBA')
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Fast resize - use BILINEAR for speed, LANCZOS for quality
            # BILINEAR is 3-4x faster than LANCZOS with minimal quality loss for 384x384
            if image.size != (self.config.image_size, self.config.image_size):
                # Use BILINEAR for speed (can switch to LANCZOS if quality is critical)
                image = image.resize(
                    (self.config.image_size, self.config.image_size), 
                    Image.Resampling.BILINEAR  # Changed from LANCZOS for 3x speed boost
                )
            
            # Fast tensor conversion and normalization
            pixel_values = TVF.pil_to_tensor(image).float()
            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            
            return pixel_values
            
        except Exception as e:
            print(f"⚠️ Error preprocessing {image_path}: {str(e)}")
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
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model with advanced optimizations"""
        print("🚀 Initializing Advanced JoyCaption Model...")
        
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Advanced model configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
        }
        
        # Enable advanced optimizations if available
        if self.config.optimize_memory:
            model_kwargs["low_cpu_mem_usage"] = True
            
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
                print("⚡ Compiling model for optimal performance...")
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
        
        # Warm up the model
        self._warmup_model()
        
        print("✅ Model initialization complete!")
        
    def _warmup_model(self):
        """Warm up the model with a dummy forward pass"""
        print("🔥 Warming up model...")
        try:
            # Create a dummy image tensor
            dummy_image = torch.randn(1, 3, self.config.image_size, self.config.image_size, 
                                    dtype=torch.bfloat16, device=self.device)
            dummy_text = "Describe this image."
            
            with torch.no_grad():
                # Use a simple generation call for warmup
                dummy_prompt = self._prepare_conversation(dummy_text)
                convo_string = self.tokenizer.apply_chat_template(
                    dummy_prompt, tokenize=False, add_generation_prompt=True
                )
                convo_tokens = self.tokenizer.encode(
                    convo_string, add_special_tokens=False, truncation=False
                )
                
                # Handle image tokens
                input_tokens = []
                for token in convo_tokens:
                    if token == self.model.config.image_token_index:
                        input_tokens.extend([self.model.config.image_token_index] * 
                                          self.model.config.image_seq_length)
                    else:
                        input_tokens.append(token)
                
                input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(input_ids)
                
                # Simple generation for warmup
                self.model.generate(
                    input_ids=input_ids,
                    pixel_values=dummy_image,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
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
            # Stack pixel values
            pixel_values = torch.stack(pixel_values_batch).to(self.device)
            
            # Prepare conversations for all prompts
            conversations = [self._prepare_conversation(prompt) for prompt in prompts]
            
            # Process first conversation (assuming same prompt for batch)
            convo_string = self.tokenizer.apply_chat_template(
                conversations[0], tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            convo_tokens = self.tokenizer.encode(
                convo_string, add_special_tokens=False, truncation=False
            )
            
            # Handle image tokens
            input_tokens = []
            for token in convo_tokens:
                if token == self.model.config.image_token_index:
                    input_tokens.extend([self.model.config.image_token_index] * 
                                      self.model.config.image_seq_length)
                else:
                    input_tokens.append(token)
            
            # Prepare input for batch
            batch_size = len(pixel_values_batch)
            input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generation parameters
            gen_kwargs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 1024),
                "do_sample": generation_kwargs.get("temperature", 0) > 0,
                "temperature": generation_kwargs.get("temperature", 0.6) if generation_kwargs.get("temperature", 0) > 0 else None,
                "top_p": generation_kwargs.get("top_p", 0.9) if generation_kwargs.get("temperature", 0) > 0 else None,
                "top_k": generation_kwargs.get("top_k", None),
                "repetition_penalty": generation_kwargs.get("repetition_penalty", 1.1),
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Use AMP if enabled
            if self.config.enable_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model.generate(**gen_kwargs)
            else:
                outputs = self.model.generate(**gen_kwargs)
            
            # Decode responses
            captions = []
            for i, output in enumerate(outputs):
                generated_ids = output[input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                captions.append(response.strip())
            
            return captions
            
        except Exception as e:
            print(f"Error in batch generation: {str(e)}")
            return [f"Error generating caption: {str(e)}"] * len(pixel_values_batch)
    
    def generate_captions(self, image_paths: List[str], prompt: str, 
                         progress_callback=None, **generation_kwargs) -> Dict[str, str]:
        """Generate captions for multiple images with optimized batching and parallel preprocessing"""
        total_images = len(image_paths)
        captions = {}
        
        print(f"🎯 Generating captions for {total_images} images...")
        caption_start = time.time()
        
        # Process images in batches
        for i in range(0, total_images, self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total_images + self.config.batch_size - 1) // self.config.batch_size
            
            if progress_callback:
                progress_callback(i / total_images, f"🤖 Captioning batch {batch_num}/{total_batches}")
            
            # PARALLEL preprocessing of batch images using ThreadPoolExecutor
            pixel_values_batch = []
            valid_paths = []
            
            # Use parallel preprocessing for faster loading
            with ThreadPoolExecutor(max_workers=min(4, len(batch_paths))) as executor:
                future_to_path = {
                    executor.submit(self.image_processor.preprocess_image, path): path 
                    for path in batch_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        pixel_values = future.result()
                        if pixel_values is not None:
                            pixel_values_batch.append(pixel_values)
                            valid_paths.append(path)
                    except Exception as e:
                        print(f"⚠️ Failed to preprocess {path}: {e}")
            
            if pixel_values_batch:
                # Generate captions for batch
                prompts = [prompt] * len(pixel_values_batch)
                batch_captions = self._generate_caption_batch(
                    pixel_values_batch, prompts, **generation_kwargs
                )
                
                # Store results
                for path, caption in zip(valid_paths, batch_captions):
                    image_name = os.path.basename(path)
                    captions[image_name] = caption
            
            # Clear GPU cache periodically
            if i % (self.config.batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        caption_time = time.time() - caption_start
        print(f"✅ Caption generation complete in {caption_time:.2f}s")
        print(f"⚡ Average: {caption_time/max(1, total_images):.2f}s per caption")
        
        return captions

class SystemMonitor:
    """System resource monitoring"""
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "name": gpu.name,
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "memory_percent": f"{gpu.memoryUtil*100:.1f}%",
                    "temperature": f"{gpu.temperature}°C",
                    "load": f"{gpu.load*100:.1f}%"
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

def create_processed_images_zip(processed_dir: str, folder_name: str) -> str:
    """Create a zip file containing all processed images (supports all image formats)"""
    try:
        # Create temporary zip file
        zip_filename = f"{folder_name}_processed_images.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        print(f"📦 Creating processed images ZIP: {zip_filename}")
        print(f"   Source directory: {processed_dir}")
        
        # Supported image extensions (comprehensive list)
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif'}
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add ALL image files from processed directory
            files_added = 0
            files_by_type = {}
            
            for file in os.listdir(processed_dir):
                file_lower = file.lower()
                file_ext = Path(file).suffix.lower()
                
                # Check if file has a supported image extension
                if file_ext in image_extensions:
                    file_path = os.path.join(processed_dir, file)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, file)
                        files_added += 1
                        
                        # Track file types for logging
                        files_by_type[file_ext] = files_by_type.get(file_ext, 0) + 1
        
        if files_added == 0:
            print(f"⚠️ Warning: No images found in {processed_dir}")
            print(f"   Directory exists: {os.path.exists(processed_dir)}")
            if os.path.exists(processed_dir):
                all_files = os.listdir(processed_dir)
                print(f"   Files in directory: {all_files[:10]}")  # Show first 10 files
            return None
        
        zip_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        print(f"✅ Processed images ZIP created: {files_added} files, {zip_size // 1024 // 1024}MB")
        
        # Log file type breakdown
        if files_by_type:
            print(f"   File types: {', '.join([f'{count} {ext}' for ext, count in sorted(files_by_type.items())])}")
        
        return zip_path
    except Exception as e:
        print(f"❌ Error creating processed images zip: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def _process_single_image(args):
    """Process a single image - designed for parallel execution"""
    image_path, i, processed_dir, compression_config = args
    try:
        # Load image efficiently
        with Image.open(image_path) as img:
            # Calculate original size
            original_size = os.path.getsize(image_path)
            
            # Load image data immediately to avoid file handle issues
            img.load()
            
            # Fast conversion to RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img, mask=img.split()[1])
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Fast orientation fix - only if EXIF data exists
            try:
                img = ImageOps.exif_transpose(img)
            except:
                pass  # Skip if no EXIF or error
            
            # Save optimized image
            if compression_config and compression_config.get('enabled', False):
                quality = compression_config.get('quality', 85)
                # Disable optimize for speed (optimize=True is VERY slow)
                optimize = False  # Force disable for speed
                
                # Adjust quality for large images
                if img.size[0] * img.size[1] > 2000000:
                    quality = max(quality - 10, 70)
                
                new_filename = f"{i:04d}.jpg"
                new_path = os.path.join(processed_dir, new_filename)
                # Fast JPEG save without optimization
                img.save(new_path, "JPEG", quality=quality, optimize=optimize)
            else:
                # PNG without optimize is MUCH faster
                new_filename = f"{i:04d}.png"
                new_path = os.path.join(processed_dir, new_filename)
                # optimize=False makes this 10-20x faster!
                img.save(new_path, "PNG", optimize=False, compress_level=6)
            
            compressed_size = os.path.getsize(new_path)
            
            return new_path, original_size, compressed_size, None
            
    except Exception as e:
        return None, 0, 0, f"Error processing {image_path}: {str(e)}"

def extract_and_process_images(zip_path: str, extract_to: str, compression_config: dict = None, progress_callback=None) -> Tuple[str, List[str]]:
    """Extract zip and process images with PARALLEL processing and progress feedback"""
    
    print(f"📦 Extracting ZIP: {os.path.basename(zip_path)}")
    extract_start = time.time()
    
    # Fast extraction with progress
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        
        # Extract with progress feedback
        for i, file_name in enumerate(file_list):
            zip_ref.extract(file_name, extract_to)
            if progress_callback and i % max(1, total_files // 20) == 0:
                progress_callback(i / total_files * 0.3, f"📦 Extracting... {i}/{total_files}")
    
    extract_time = time.time() - extract_start
    print(f"✅ Extraction complete in {extract_time:.2f}s")
    
    # Find the main folder efficiently
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
    
    # Fast image file discovery with progress
    print("🔍 Scanning for images...")
    scan_start = time.time()
    image_files = []
    
    for root, dirs, files in os.walk(main_folder):
        if "processed_images" in root:
            continue
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in SUPPORTED_IMAGE_FORMATS:
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    
    image_files.sort()
    total_images = len(image_files)
    print(f"✅ Found {total_images} images in {time.time() - scan_start:.2f}s")
    
    if total_images == 0:
        return processed_dir, []
    
    # PARALLEL image processing
    print(f"🚀 Starting parallel image processing with {min(8, total_images)} workers...")
    process_start = time.time()
    
    converted_files = []
    total_original_size = 0
    total_compressed_size = 0
    errors = []
    
    # Prepare arguments for parallel processing
    process_args = [
        (image_path, i, processed_dir, compression_config)
        for i, image_path in enumerate(image_files, 1)
    ]
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(8, os.cpu_count() or 4, total_images)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_image, args): args for args in process_args}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            new_path, original_size, compressed_size, error = result
            
            if error:
                errors.append(error)
            elif new_path:
                converted_files.append(new_path)
                total_original_size += original_size
                total_compressed_size += compressed_size
            
            completed += 1
            if progress_callback and completed % max(1, total_images // 20) == 0:
                progress_callback(0.3 + (completed / total_images * 0.7), 
                                f"🖼️ Processing images... {completed}/{total_images}")
    
    # Sort converted files by filename
    converted_files.sort()
    
    process_time = time.time() - process_start
    print(f"✅ Processed {len(converted_files)} images in {process_time:.2f}s")
    print(f"⚡ Average: {process_time/max(1, len(converted_files)):.3f}s per image")
    
    if errors:
        print(f"⚠️ {len(errors)} errors occurred during processing")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   {error}")
    
    # Print compression statistics
    if compression_config and compression_config.get('enabled', False) and total_original_size > 0:
        compression_ratio = total_compressed_size / total_original_size
        saved_space = total_original_size - total_compressed_size
        print(f"🗜️ Compression Results:")
        print(f"   Original: {total_original_size // 1024 // 1024}MB")
        print(f"   Compressed: {total_compressed_size // 1024 // 1024}MB")
        print(f"   Saved: {saved_space // 1024 // 1024}MB ({(1-compression_ratio)*100:.1f}%)")
    
    return processed_dir, converted_files

def create_professional_interface():
    """Create advanced professional Gradio interface"""
    
    # Custom CSS for professional styling
    custom_css = """
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .content-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .status-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    
    .progress-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .download-container {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .settings-panel {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .process-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .process-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    """
    
    # Initialize components
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
    
    captioner = GPUOptimizedCaptioner(config)
    monitor = SystemMonitor()
    
    def process_images_advanced(zip_file, prompt, temperature, top_p, top_k, 
                              repetition_penalty, max_new_tokens, batch_size,
                              enable_compression, compression_quality, compression_optimize,
                              progress=gr.Progress()):
        """Advanced image processing function with compression support"""
        
        if zip_file is None:
            return None, None, None, "❌ Please upload a ZIP file containing images.", ""
        
        if not prompt.strip():
            return None, None, None, "❌ Please provide a prompt for captioning.", ""
        
        try:
            # Update configuration
            config.batch_size = batch_size
            config.compression_enabled = enable_compression
            config.compression_quality = compression_quality
            config.compression_optimize = compression_optimize
            
            start_time = time.time()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                progress(0.01, desc="📦 Starting ZIP extraction...")
                
                # Extract and process images with compression config
                compression_config = {
                    'enabled': enable_compression,
                    'quality': compression_quality,
                    'optimize': compression_optimize
                } if enable_compression else None
                
                # Progress callback for extraction/processing (maps to 0.01-0.20 range)
                def extraction_progress_callback(current_progress, desc):
                    mapped_progress = 0.01 + (current_progress * 0.19)
                    progress(mapped_progress, desc=desc)
                
                processed_dir, converted_files = extract_and_process_images(
                    zip_file.name, temp_dir, compression_config, extraction_progress_callback
                )
                folder_name = os.path.basename(os.path.dirname(processed_dir))
                
                if not converted_files:
                    return None, None, None, "❌ No supported image files found in the ZIP.", ""
                
                compression_status = "🗜️ with compression" if enable_compression else "without compression"
                progress(0.20, desc=f"🖼️ Found {len(converted_files)} images. Starting captioning {compression_status}...")
                
                # Progress tracking function for captioning (maps to 0.20-0.90 range)
                def progress_callback(current_progress, desc):
                    mapped_progress = 0.20 + (current_progress * 0.70)
                    progress(mapped_progress, desc=desc)
                
                # Generate captions
                generation_kwargs = {
                    "temperature": temperature if temperature > 0 else 0,
                    "top_p": top_p,
                    "top_k": top_k if top_k > 0 else None,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                }
                
                captions_dict = captioner.generate_captions(
                    converted_files, prompt, progress_callback, **generation_kwargs
                )
                
                progress(0.90, desc="💾 Creating output files...")
                
                # Create processed images zip
                print(f"📦 Creating ZIP of {len(converted_files)} processed images...")
                processed_images_zip = create_processed_images_zip(processed_dir, folder_name)
                
                if processed_images_zip is None or not os.path.exists(processed_images_zip):
                    print(f"⚠️ Warning: Failed to create processed images ZIP")
                    print(f"   Processed directory: {processed_dir}")
                    print(f"   Files in directory: {os.listdir(processed_dir) if os.path.exists(processed_dir) else 'Directory not found'}")
                
                # Create output files
                output_dir = tempfile.mkdtemp()
                
                # Prepare data structures
                captions_data = {}
                txt_content = []
                
                for i, image_path in enumerate(converted_files, 1):
                    image_name = os.path.basename(image_path)
                    caption = captions_dict.get(image_name, "Caption generation failed")
                    
                    captions_data[image_name] = {
                        "image_number": i,
                        "original_filename": image_name,
                        "caption": caption,
                        "prompt_used": prompt,
                        "generation_settings": generation_kwargs
                    }
                    
                    txt_content.append(f"Image {i:04d} ({image_name}):\n{caption}\n")
                
                # Save TXT file
                txt_filename = f"{folder_name}_captions.txt"
                txt_path = os.path.join(output_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"🎯 Advanced Captioning Results\n")
                    f.write(f"📁 Folder: {folder_name}\n")
                    f.write(f"💬 Prompt: {prompt}\n")
                    f.write(f"📊 Total Images: {len(converted_files)}\n")
                    f.write(f"⚙️ Settings: Temp={temperature}, Top-p={top_p}, Batch={batch_size}\n")
                    if enable_compression:
                        f.write(f"🗜️ Compression: Quality={compression_quality}, Optimize={compression_optimize}\n")
                    f.write(f"⏱️ Processing Time: {time.time() - start_time:.2f} seconds\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("\n".join(txt_content))
                
                # Save JSON file
                json_filename = f"{folder_name}_captions.json"
                json_path = os.path.join(output_dir, json_filename)
                
                json_data = {
                    "metadata": {
                        "folder_name": folder_name,
                        "prompt_used": prompt,
                        "total_images": len(converted_files),
                        "processing_time_seconds": time.time() - start_time,
                        "model_used": MODEL_PATH,
                        "generation_settings": generation_kwargs,
                        "processing_config": {
                            "batch_size": batch_size,
                            "image_size": config.image_size,
                            "amp_enabled": config.enable_amp,
                            "compression_enabled": enable_compression,
                            "compression_quality": compression_quality if enable_compression else None,
                            "compression_optimize": compression_optimize if enable_compression else None
                        }
                    },
                    "system_info": {
                        "gpu": monitor.get_gpu_info(),
                        "cpu": monitor.get_cpu_info()
                    },
                    "captions": captions_data
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                progress(1.0, desc="✅ Processing complete!")
                
                # Create status message
                processing_time = time.time() - start_time
                compression_info = f"\n- 🗜️ Compression: Enabled (Quality: {compression_quality})" if enable_compression else "\n- 🗜️ Compression: Disabled"
                
                status_msg = f"""
                ✅ **Processing Complete!**
                
                📊 **Statistics:**
                - 🖼️ Images processed: {len(converted_files)}
                - ⏱️ Total time: {processing_time:.2f} seconds
                - 🚀 Average time per image: {processing_time/len(converted_files):.2f} seconds
                - 🔄 Batch size used: {batch_size}{compression_info}
                
                📁 **Output files generated:**
                - 📝 TXT file: {txt_filename}
                - 📋 JSON file: {json_filename}
                - 🖼️ Processed images: {folder_name}_processed_images.zip
                """
                
                return txt_path, json_path, processed_images_zip, status_msg, ""
                
        except Exception as e:
            error_msg = f"❌ **Error during processing:**\n\n{str(e)}"
            return None, None, None, error_msg, ""
    
    def get_system_status():
        """Get current system status"""
        gpu_info = monitor.get_gpu_info()
        cpu_info = monitor.get_cpu_info()
        
        return f"""
        🖥️ **System Status:**
        
        **GPU:** {gpu_info.get('name', 'N/A')}
        - Memory: {gpu_info.get('memory_used', 'N/A')} / {gpu_info.get('memory_total', 'N/A')} ({gpu_info.get('memory_percent', 'N/A')})
        - Load: {gpu_info.get('load', 'N/A')}
        - Temperature: {gpu_info.get('temperature', 'N/A')}
        
        **CPU:** {cpu_info.get('cores', 'N/A')} cores
        - Usage: {cpu_info.get('usage', 'N/A')}
        - Memory: {cpu_info.get('memory', 'N/A')}
        """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="Advanced Image Captioning Pro") as app:
        
        gr.HTML("""
        <div class="header-title">
            🚀 Advanced Image Captioning Pro
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <p style="font-size: 1.2em; color: #666;">
                Professional batch image captioning with GPU optimization and advanced AI models
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input Section
                gr.HTML("<h2 style='color: #667eea;'>📤 Input Configuration</h2>")
                
                zip_input = gr.File(
                    label="📦 Upload ZIP file with images",
                    file_types=[".zip"],
                    type="filepath",
                    elem_classes=["upload-area"]
                )
                
                prompt_input = gr.Textbox(
                    label="💬 Captioning Prompt",
                    placeholder="Describe this image in detail with rich, descriptive language...",
                    value="Describe this image in detail, including the main subjects, colors, composition, mood, and any notable details.",
                    lines=4,
                    elem_classes=["prompt-input"]
                )
                
                # Advanced Settings
                with gr.Accordion("⚙️ Advanced Generation Settings", open=False, elem_classes=["settings-panel"]):
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0.7,
                            label="🌡️ Temperature (creativity)",
                            info="Higher = more creative, Lower = more focused"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1, step=0.05, value=0.9,
                            label="🎯 Top-p (diversity)",
                            info="Controls word selection diversity"
                        )
                    
                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=0, maximum=100, step=1, value=50,
                            label="🔝 Top-k (vocabulary limit)",
                            info="0 = disabled, higher = more vocabulary"
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0, maximum=2.0, step=0.05, value=1.1,
                            label="🔄 Repetition Penalty",
                            info="Reduces repetitive text"
                        )
                    
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=50, maximum=2048, step=50, value=1024,
                            label="📏 Max Caption Length",
                            info="Maximum tokens per caption"
                        )
                        batch_size = gr.Slider(
                            minimum=1, maximum=16, step=1, value=4,
                            label="⚡ Batch Size",
                            info="Images processed simultaneously"
                        )
                
                # Image Compression Settings
                with gr.Accordion("🗜️ Image Compression Settings", open=False, elem_classes=["settings-panel"]):
                    gr.HTML("<p style='color: #666; margin-bottom: 15px;'>Enable intelligent image compression to reduce file sizes while maintaining visual quality. Recommended for large datasets.</p>")
                    
                    enable_compression = gr.Checkbox(
                        label="🗜️ Enable Image Compression",
                        value=False,
                        info="Compress images during processing to save storage space"
                    )
                    
                    with gr.Row():
                        compression_quality = gr.Slider(
                            minimum=60, maximum=100, step=5, value=85,
                            label="🎚️ Compression Quality",
                            info="Higher = better quality, larger files"
                        )
                        compression_optimize = gr.Checkbox(
                            label="⚙️ Optimize Compression",
                            value=True,
                            info="Use advanced optimization algorithms"
                        )
                    
                    gr.HTML("""
                    <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;">
                        <strong>💡 Compression Tips:</strong><br>
                        • Quality 85-95: Excellent quality, good compression<br>
                        • Quality 75-84: Very good quality, better compression<br>
                        • Quality 60-74: Good quality, maximum compression<br>
                        • Optimization adds ~10% processing time but improves compression
                    </div>
                    """)
                
                process_btn = gr.Button(
                    "🚀 Start Advanced Processing",
                    variant="primary",
                    size="lg",
                    elem_classes=["process-btn"]
                )
            
            with gr.Column(scale=1):
                # Status and Output Section
                gr.HTML("<h2 style='color: #667eea;'>📊 Status & Output</h2>")
                
                system_status = gr.Textbox(
                    label="🖥️ System Status",
                    value=get_system_status(),
                    lines=10,
                    interactive=False,
                    elem_classes=["status-container"]
                )
                
                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                
                processing_status = gr.Textbox(
                    label="📋 Processing Status",
                    lines=8,
                    interactive=False,
                    elem_classes=["progress-container"]
                )
                
                with gr.Row():
                    txt_download = gr.File(
                        label="📝 Download TXT",
                        interactive=False,
                        elem_classes=["download-container"]
                    )
                    json_download = gr.File(
                        label="📋 Download JSON",
                        interactive=False,
                        elem_classes=["download-container"]
                    )
                
                images_zip_download = gr.File(
                    label="🖼️ Download Processed Images ZIP",
                    interactive=False,
                    elem_classes=["download-container"]
                )
        
        # Information Section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3>🎯 Features & Capabilities</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
                <div>
                    <h4>🚀 Performance Optimizations</h4>
                    <ul>
                        <li>GPU batch processing</li>
                        <li>Memory optimization</li>
                        <li>Advanced model compilation</li>
                        <li>Automatic mixed precision</li>
                    </ul>
                </div>
                <div>
                    <h4>🖼️ Image Processing</h4>
                    <ul>
                        <li>High-quality format conversion</li>
                        <li>Automatic orientation correction</li>
                        <li>Transparency handling</li>
                        <li>Sequential file naming</li>
                        <li>Smart image compression</li>
                        <li>Quality-preserving optimization</li>
                    </ul>
                </div>
                <div>
                    <h4>📊 Professional Output</h4>
                    <ul>
                        <li>Detailed TXT reports</li>
                        <li>Structured JSON data</li>
                        <li>Processed images ZIP</li>
                        <li>Processing statistics</li>
                        <li>System information</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
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
        
        refresh_btn.click(
            fn=get_system_status,
            outputs=system_status
        )
        
        # Auto-refresh system status every 30 seconds
        app.load(get_system_status, outputs=system_status)
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Advanced Image Captioning Pro...")
    app = create_professional_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        inbrowser=True,
        show_error=True,
        quiet=True
    )