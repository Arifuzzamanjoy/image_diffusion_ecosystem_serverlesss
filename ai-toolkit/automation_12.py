import os
import torch
import random
import json
import time
import tempfile
import zipfile
import shutil
import traceback
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import gradio as gr
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import queue
import uuid


# Application Configuration
APP_VERSION = "2.0.0 Pro"
APP_NAME = "FLUX Studio Professional"
LICENSE_TYPE = "Commercial License"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flux_studio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Usage tracking
class UsageTracker:
    def __init__(self):
        self.session_start = time.time()
        self.images_generated = 0
        self.total_generation_time = 0
        self.session_id = str(uuid.uuid4())[:8]
        
    def track_generation(self, generation_time: float, num_images: int = 1):
        self.images_generated += num_images
        self.total_generation_time += generation_time
        
    def get_stats(self) -> Dict:
        session_duration = time.time() - self.session_start
        avg_time_per_image = self.total_generation_time / max(1, self.images_generated)
        
        return {
            "session_id": self.session_id,
            "session_duration": session_duration,
            "images_generated": self.images_generated,
            "total_generation_time": self.total_generation_time,
            "avg_time_per_image": avg_time_per_image,
            "images_per_minute": (self.images_generated / (session_duration / 60)) if session_duration > 0 else 0
        }

# Global usage tracker
usage_tracker = UsageTracker()

# Enhanced error handling
class FluxStudioError(Exception):
    """Custom exception for FLUX Studio errors"""
    pass

def handle_errors(func):
    """Decorator for consistent error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FluxStudioError(error_msg)
    return wrapper

# System monitoring
def get_system_stats() -> Dict:
    """Get current system resource usage"""
    try:
        memory = psutil.virtual_memory()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        gpu_used = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "gpu_memory_total_gb": gpu_memory / (1024**3),
            "gpu_memory_used_gb": gpu_used / (1024**3),
            "gpu_memory_percent": (gpu_used / gpu_memory * 100) if gpu_memory > 0 else 0
        }
    except Exception as e:
        logger.warning(f"Could not get system stats: {e}")
        return {"error": str(e)}

# Set up optimal cache location for runpod
CACHE_DIR = os.path.expanduser("/root/.cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure environment variables for better caching on runpod
os.environ.update({
    "HF_HOME": CACHE_DIR,
    "TRANSFORMERS_CACHE": CACHE_DIR,
    "HUGGINGFACE_HUB_CACHE": CACHE_DIR,
    "SAFETENSORS_CACHE_DIR": CACHE_DIR,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
})

# Login to Huggingface
load_dotenv()  # Loads variables from .env

hf_token = os.getenv('HF_TOKEN')
if hf_token:
    # Suppress warnings by setting add_to_git_credential=False
    login(token=hf_token, add_to_git_credential=False)
else:
    print("⚠️ HF_TOKEN not found in .env file. Please add it for model downloads.")

# Free up CUDA memory first
torch.cuda.empty_cache()

# Use bfloat16 for better performance on newer GPUs
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading models, please wait...")



# Load required models
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14-336", 
    torch_dtype=dtype, 
    cache_dir=os.environ["HF_HOME"]
).to(device)

#openai/clip-vit-large-patch14
#openai/clip-vit-large-patch14-336
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14-336", 
    cache_dir=os.environ["HF_HOME"]
)



vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="vae", 
    torch_dtype=dtype, 
    cache_dir=os.environ["HF_HOME"]
).to(device)

#vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype, cache_dir=os.environ["HF_HOME"]).to(device)



# Load T5 models for enhanced text understanding
text_encoder_2 = T5EncoderModel.from_pretrained(
    "google/t5-v1_1-xxl", 
    torch_dtype=dtype, 
    cache_dir=os.environ["HF_HOME"]
).to(device)

tokenizer_2 = T5TokenizerFast.from_pretrained(
    "google/t5-v1_1-xxl", 
    legacy=True, 
    cache_dir=os.environ["HF_HOME"]
)


scheduler = FlowMatchEulerDiscreteScheduler()

print("🔄 Loading the pipeline, please wait...")

#black-forest-labs/FLUX.1-dev
#John6666/nsfw-master-flux-lora-merged-with-flux1-dev-fp16-v10-fp8-flux
# Initialize the pipeline with runpod optimized settings
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    torch_dtype=dtype,
    scheduler=scheduler,
    cache_dir=os.environ["HF_HOME"]
)



###############


####################
# Move to GPU
pipe.to(device)

# Optimize memory usage
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✅ Using xformers for memory efficient attention")
except Exception:
    print("⚠️ xformers not available. Using default attention mechanism.")

print("✅ Model loaded and ready")

# Watermark and branding functionality
def add_watermark(image: Image.Image, text: str = f"{APP_NAME} v{APP_VERSION}") -> Image.Image:
    """Add watermark to generated images"""
    try:
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        
        # Try to use a better font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position watermark in bottom right corner
        x = image.width - text_width - 10
        y = image.height - text_height - 10
        
        # Add semi-transparent background
        draw.rectangle([x-5, y-2, x+text_width+5, y+text_height+2], fill=(0, 0, 0, 128))
        
        # Add text
        draw.text((x, y), text, fill=(255, 255, 255, 200), font=font)
        
        return watermarked
    except Exception as e:
        logger.warning(f"Could not add watermark: {e}")
        return image

# Queue management for batch processing
class GenerationQueue:
    def __init__(self, max_size: int = 100):
        self.queue = queue.Queue(maxsize=max_size)
        self.active_jobs = {}
        self.completed_jobs = {}
        
    def add_job(self, job_id: str, job_data: Dict) -> bool:
        """Add a job to the queue"""
        try:
            self.queue.put((job_id, job_data), timeout=1)
            self.active_jobs[job_id] = {
                "status": "queued",
                "created_at": time.time(),
                "data": job_data
            }
            return True
        except queue.Full:
            return False
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a specific job"""
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        elif job_id in self.active_jobs:
            return self.active_jobs[job_id]
        else:
            return {"status": "not_found"}
    
    def complete_job(self, job_id: str, result: Dict):
        """Mark job as completed"""
        if job_id in self.active_jobs:
            self.completed_jobs[job_id] = {
                "status": "completed",
                "completed_at": time.time(),
                "result": result
            }
            del self.active_jobs[job_id]

# Global queue manager
generation_queue = GenerationQueue()

# Enhanced validation functions
def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate user prompt"""
    if not prompt or len(prompt.strip()) < 3:
        return False, "Prompt must be at least 3 characters long"
    
    if len(prompt) > 2000:
        return False, "Prompt is too long (max 2000 characters)"
    
    # Check for potentially harmful content (basic filtering)
    harmful_keywords = ['virus', 'hack', 'exploit', 'malware']
    if any(keyword in prompt.lower() for keyword in harmful_keywords):
        return False, "Prompt contains potentially harmful content"
    
    return True, "Valid"

def validate_generation_params(width: int, height: int, steps: int, guidance_scale: float) -> Tuple[bool, str]:
    """Validate generation parameters"""
    if width < 256 or width > 2048 or height < 256 or height > 2048:
        return False, "Image dimensions must be between 256 and 2048 pixels"
    
    if width % 64 != 0 or height % 64 != 0:
        return False, "Image dimensions must be divisible by 64"
    
    if steps < 10 or steps > 150:
        return False, "Steps must be between 10 and 150"
    
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        return False, "Guidance scale must be between 1.0 and 20.0"
    
    return True, "Valid"

# Progress tracking
class ProgressTracker:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.current_image = 0
        self.total_images = 0
        self.start_time = None
        
    def start(self, total_images: int, steps_per_image: int):
        self.current_step = 0
        self.total_steps = steps_per_image
        self.current_image = 0
        self.total_images = total_images
        self.start_time = time.time()
        
    def update_step(self, step: int):
        self.current_step = step
        
    def next_image(self):
        self.current_image += 1
        self.current_step = 0
        
    def get_progress(self) -> Dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        overall_progress = ((self.current_image * self.total_steps + self.current_step) / 
                          (self.total_images * self.total_steps)) if self.total_steps > 0 else 0
        
        return {
            "current_image": self.current_image,
            "total_images": self.total_images,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "overall_progress": overall_progress,
            "elapsed_time": elapsed,
            "eta": (elapsed / overall_progress - elapsed) if overall_progress > 0 else 0
        }

# Global progress tracker
progress_tracker = ProgressTracker()

# Helper functions for image saving
def save_images_to_files(images):
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for i, image in enumerate(images):
        file_path = os.path.join(temp_dir, f"generated_image_{i}.png")
        image.save(file_path)
        saved_paths.append(file_path)
    
    return saved_paths

# Save images to a specific directory
def save_images_to_directory(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, image in enumerate(images):
        timestamp = int(time.time())
        file_path = os.path.join(output_dir, f"generated_image_{timestamp}_{i}.png")
        image.save(file_path)
        saved_paths.append(file_path)
    
    return saved_paths

# New function to parse uploaded prompt files
def parse_prompt_file(uploaded_file):
    """Parse uploaded JSON or text file to extract prompts"""
    try:
        file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as JSON first
        if file_path.lower().endswith('.json'):
            try:
                # Clean up common JSON formatting issues
                content = content.replace('\n', ' ').replace('\r', ' ')
                content = ' '.join(content.split())  # Normalize whitespace
                
                data = json.loads(content)
                if isinstance(data, list):
                    prompts = []
                    for item in data:
                        if isinstance(item, dict):
                            # Handle various JSON structures
                            if 'prompt' in item:
                                prompt_text = item['prompt'].strip()
                                if prompt_text:
                                    prompts.append(prompt_text)
                            elif 'text' in item:
                                prompt_text = item['text'].strip()
                                if prompt_text:
                                    prompts.append(prompt_text)
                        elif isinstance(item, str):
                            prompt_text = item.strip()
                            if prompt_text:
                                prompts.append(prompt_text)
                    return prompts
                elif isinstance(data, dict):
                    if 'prompts' in data:
                        return [p.strip() for p in data['prompts'] if p.strip()]
                    elif 'prompt' in data:
                        return [data['prompt'].strip()] if data['prompt'].strip() else []
                else:
                    return [str(data).strip()] if str(data).strip() else []
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Content preview: {content[:200]}...")
                # If JSON parsing fails, treat as text
                pass
        
        # Parse as text file (each line is a prompt)
        prompts = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                prompts.append(line)
        return prompts
        
    except Exception as e:
        print(f"Error parsing prompt file: {e}")
        return []

# New function to create ZIP of batch output
def create_batch_zip(output_dir, batch_name=None):
    """Create a ZIP file containing all images from batch processing"""
    try:
        if not os.path.exists(output_dir):
            print(f"❌ Output directory does not exist: {output_dir}")
            return None
        
        if batch_name is None:
            batch_name = f"batch_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create ZIP in temp directory for Gradio compatibility
        zip_path = os.path.join("/tmp", f"{batch_name}.zip")
        
        print(f"📦 Creating ZIP file: {zip_path}")
        
        image_count = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
                        image_count += 1
                        print(f"   Added: {arcname}")
                    elif file.endswith('.json'):
                        # Also include metadata files
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
                        print(f"   Added: {arcname}")
        
        if image_count > 0:
            print(f"✅ ZIP created successfully with {image_count} images: {zip_path}")
            return zip_path
        else:
            print("❌ No images found to add to ZIP")
            return None
        
    except Exception as e:
        print(f"❌ Error creating batch ZIP: {e}")
        traceback.print_exc()
        return None

# Enhanced image generation with progress tracking and validation
@handle_errors
def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    seed,
    lora_path,
    lora_scale=0.8,
    num_images=1,
    add_watermark_flag=False,
    progress_callback=None
):
    # Validate inputs
    prompt_valid, prompt_msg = validate_prompt(prompt)
    if not prompt_valid:
        raise FluxStudioError(f"Invalid prompt: {prompt_msg}")
    
    params_valid, params_msg = validate_generation_params(width, height, num_inference_steps, guidance_scale)
    if not params_valid:
        raise FluxStudioError(f"Invalid parameters: {params_msg}")
    
    start_time = time.time()
    progress_tracker.start(num_images, num_inference_steps)
    
    # Unload any previous LoRA weights
    if hasattr(pipe, 'unload_lora_weights'):
        try:
            pipe.unload_lora_weights()
        except:
            pass
    
    # Load LoRA weights if specified
    if lora_path and lora_path.strip():
        try:
            pipe.load_lora_weights(lora_path)
            print(f"✅ Loaded LoRA weights from {lora_path}")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights: {e}")
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Configure optimal parameters for runpod
    safe_width = (width // 16) * 16  # Ensure width is divisible by 16
    safe_height = (height // 16) * 16  # Ensure height is divisible by 16
    safe_guidance = min(guidance_scale, 12.0)
    
    # Enhanced prompt for better results
    full_prompt = f"{prompt} "
    
    print(f"🔄 Generating image with seed: {seed}")
    
    try:
        # Direct approach using prompt without embedding
        images = pipe(
            prompt=full_prompt,
            prompt_2=full_prompt, 
            negative_prompt=negative_prompt if negative_prompt else None,
            height=safe_height,
            width=safe_width,
            guidance_scale=safe_guidance,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=num_images,
            joint_attention_kwargs={"scale": lora_scale} if lora_path.strip() else None,
         #   joint_attention_kwargs=joint_attention_kwargs,
            output_type="pil"
        ).images
        
        if images:
            # Add watermarks if enabled
            if add_watermark_flag:
                watermarked_images = [add_watermark(img) for img in images]
                images = watermarked_images
            
            saved_paths = save_images_to_files(images)
            
            # Track usage
            generation_time = time.time() - start_time
            usage_tracker.track_generation(generation_time, len(images))
            
            # Log successful generation
            logger.info(f"Successfully generated {len(images)} images in {generation_time:.2f}s")
            
            return images, seed, saved_paths
        else:
            return [], seed, []
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        # Try fallback with safer settings
        try:
            fallback_images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=3.0,
                generator=generator,
                num_images_per_prompt=1,
            ).images
            
            if fallback_images:
                saved_paths = save_images_to_files(fallback_images)
                return fallback_images, seed, saved_paths
        except Exception as fallback_e:
            print(f"❌ Fallback generation also failed: {fallback_e}")
        
        return [], seed, []

# Cache clearing function
def clear_output_directory(output_dir):
    """Clear the output directory before starting new batch generation"""
    try:
        if os.path.exists(output_dir):
            print(f"🗑️ Clearing cache directory: {output_dir}")
            shutil.rmtree(output_dir)
            print(f"✅ Cache cleared successfully")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created fresh output directory: {output_dir}")
    except Exception as e:
        print(f"⚠️ Error clearing cache: {e}")

# Enhanced batch image generation with file upload support
def generate_images_from_file(
    uploaded_file,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    lora_path,
    lora_scale,
    output_dir,
    create_zip=True
):
    """Generate images from uploaded JSON or text file containing prompts"""
    try:
        # Parse prompts from uploaded file
        prompts = parse_prompt_file(uploaded_file)
        
        if not prompts:
            return [], "No valid prompts found in the uploaded file", [], [], None
        
        # Clear output directory cache before starting
        clear_output_directory(output_dir)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_dir = os.path.join(output_dir, f"batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Create temp directory for Gradio display (fixes path permission issue)
        temp_display_dir = tempfile.mkdtemp(prefix="flux_batch_", dir="/tmp")
        
        # Unload any previous LoRA weights
        if hasattr(pipe, 'unload_lora_weights'):
            try:
                pipe.unload_lora_weights()
            except:
                pass
        
        # Load LoRA weights if specified
        if lora_path and lora_path.strip():
            try:
                pipe.load_lora_weights(lora_path)
                print(f"✅ Loaded LoRA weights from {lora_path}")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA weights: {e}")
        
        generated_images = []
        all_image_paths = []
        temp_image_paths = []  # For Gradio display
        seeds_used = []
        
        # Save batch metadata
        metadata = {
            "batch_id": timestamp,
            "total_prompts": len(prompts),
            "settings": {
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "lora_path": lora_path,
                "lora_scale": lora_scale
            },
            "prompts": [],
            "generation_time": None
        }
        
        start_time = time.time()
        
        # Process each prompt
        for i, prompt in enumerate(prompts):
            # Generate a random seed for each image
            seed = random.randint(0, 2147483647)
            seeds_used.append(seed)
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Configure optimal parameters
            safe_width = (width // 16) * 16
            safe_height = (height // 16) * 16
            safe_guidance = min(guidance_scale, 12.0)
            
            # Enhanced prompt
            full_prompt = f"{prompt}"
            
            print(f"🔄 Generating image {i+1}/{len(prompts)} with seed: {seed}")
            
            try:
                # Generate image
                images = pipe(
                    prompt=full_prompt,
                    prompt_2=full_prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=safe_height,
                    width=safe_width,
                    guidance_scale=safe_guidance,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    num_images_per_prompt=1,
                    joint_attention_kwargs={"scale": lora_scale} if lora_path.strip() else None,
                    output_type="pil"
                ).images
                
                if images:
                    # Save each image with proper naming
                    for j, image in enumerate(images):
                        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_', '-')).strip()
                        image_filename = f"prompt_{i:03d}_{safe_prompt}_{seed}.png"
                        
                        # Save to permanent location
                        image_path = os.path.join(batch_output_dir, image_filename)
                        image.save(image_path)
                        all_image_paths.append(image_path)
                        
                        # Save to temp location for Gradio display (fixes path permission)
                        temp_image_path = os.path.join(temp_display_dir, image_filename)
                        image.save(temp_image_path)
                        temp_image_paths.append(temp_image_path)
                    
                    generated_images.extend(images)
                    
                    # Add to metadata
                    metadata["prompts"].append({
                        "index": i,
                        "prompt": prompt,
                        "seed": seed,
                        "generated": True
                    })
                    
                    # Provide update after each successful generation (use temp paths for Gradio)
                    yield generated_images, f"Generated {i+1}/{len(prompts)} images", temp_image_paths, seeds_used, None
                else:
                    metadata["prompts"].append({
                        "index": i,
                        "prompt": prompt,
                        "seed": seed,
                        "generated": False,
                        "error": "No images generated"
                    })
            
            except Exception as e:
                print(f"❌ Generation failed for prompt {i+1}: {e}")
                metadata["prompts"].append({
                    "index": i,
                    "prompt": prompt,
                    "seed": seed,
                    "generated": False,
                    "error": str(e)
                })
                continue
        
        # Save metadata
        metadata["generation_time"] = time.time() - start_time
        metadata_path = os.path.join(batch_output_dir, "batch_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP file if requested
        zip_path = None
        if create_zip and generated_images:
            zip_path = create_batch_zip(batch_output_dir, f"batch_{timestamp}")
        
        success_count = len([p for p in metadata["prompts"] if p["generated"]])
        status_msg = f"✅ Batch completed! Generated {success_count}/{len(prompts)} images successfully."
        
        # Final yield with temp paths for Gradio display
        yield generated_images, status_msg, temp_image_paths, seeds_used, zip_path
    
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return [], f"Error: {str(e)}", [], [], None

# LoRA Management Functions
LORA_OUTPUT_DIR = "/workspace/Lora_Trainer_Imgen_Flux/ai-toolkit/output"
LORA_FILTER_FILE = "lora_filter.json"

def scan_lora_models():
    """Scan the LoRA output directory for available models"""
    lora_models = {}
    
    if not os.path.exists(LORA_OUTPUT_DIR):
        return lora_models
    
    try:
        for item in os.listdir(LORA_OUTPUT_DIR):
            item_path = os.path.join(LORA_OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                # Look for .safetensors files
                safetensors_files = [f for f in os.listdir(item_path) if f.endswith('.safetensors')]
                if safetensors_files:
                    # Use the first safetensors file found
                    model_file = safetensors_files[0]
                    full_path = os.path.join(item_path, model_file)
                    lora_models[item] = {
                        'path': full_path,
                        'name': item,
                        'file': model_file,
                        'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    }
    except Exception as e:
        print(f"Error scanning LoRA models: {e}")
    
    return lora_models

def load_lora_filter():
    """Load the LoRA filter settings"""
    if os.path.exists(LORA_FILTER_FILE):
        try:
            with open(LORA_FILTER_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"hidden": [], "favorites": []}

def save_lora_filter(filter_data):
    """Save the LoRA filter settings"""
    try:
        with open(LORA_FILTER_FILE, 'w') as f:
            json.dump(filter_data, f, indent=2)
    except Exception as e:
        print(f"Error saving LoRA filter: {e}")

def get_filtered_lora_choices():
    """Get LoRA choices excluding hidden ones"""
    lora_models = scan_lora_models()
    filter_data = load_lora_filter()
    hidden = set(filter_data.get("hidden", []))
    
    choices = ["None (No LoRA)"]
    for name, info in lora_models.items():
        if name not in hidden:
            size_mb = info['size'] / (1024 * 1024)
            choice_text = f"{name} ({size_mb:.1f}MB)"
            choices.append(choice_text)
    
    return choices, lora_models

def toggle_lora_visibility(lora_name, action):
    """Toggle LoRA visibility (hide/show)"""
    filter_data = load_lora_filter()
    hidden = set(filter_data.get("hidden", []))
    
    if action == "hide" and lora_name not in hidden:
        hidden.add(lora_name)
    elif action == "show" and lora_name in hidden:
        hidden.remove(lora_name)
    
    filter_data["hidden"] = list(hidden)
    save_lora_filter(filter_data)
    
    return get_filtered_lora_choices()

def get_lora_path_from_choice(choice, lora_models):
    """Convert dropdown choice back to actual LoRA path"""
    if choice == "None (No LoRA)" or not choice:
        return ""
    
    # Extract the model name from the choice (before the size info)
    model_name = choice.split(" (")[0]
    
    if model_name in lora_models:
        return lora_models[model_name]['path']
    
    return ""

def refresh_lora_list():
    """Refresh the LoRA dropdown list"""
    choices, models = get_filtered_lora_choices()
    manage_choices = list(scan_lora_models().keys())
    return gr.update(choices=choices), gr.update(choices=manage_choices)

def handle_lora_selection(choice):
    """Handle LoRA dropdown selection"""
    _, models = get_filtered_lora_choices()
    path = get_lora_path_from_choice(choice, models)
    return path

def manage_lora_visibility(selected_lora, action):
    """Handle hiding/showing LoRAs"""
    if not selected_lora:
        return gr.update(), gr.update(), "Please select a LoRA first."
    
    if action == "hide":
        choices, _ = toggle_lora_visibility(selected_lora, "hide")
        status = f"✅ Hidden '{selected_lora}' from the list."
    else:  # show
        choices, _ = toggle_lora_visibility(selected_lora, "show") 
        status = f"✅ Restored '{selected_lora}' to the list."
    
    manage_choices = list(scan_lora_models().keys())
    return gr.update(choices=choices[0]), gr.update(choices=manage_choices), status

def generate_with_enhanced_features(
    prompt, width, height, steps, guidance_scale, seed,
    lora_dropdown, custom_lora_path, lora_scale, num_images
):
    """Enhanced wrapper function with validation, progress tracking, and professional features"""
    add_watermark_flag = False  # No watermark needed
    negative_prompt = None  # FLUX doesn't use negative prompts effectively
    try:
        # Input validation
        prompt_valid, prompt_msg = validate_prompt(prompt)
        if not prompt_valid:
            return [], -1, [], f"❌ Validation Error: {prompt_msg}", "", "Ready for next generation"
        
        params_valid, params_msg = validate_generation_params(width, height, steps, guidance_scale)
        if not params_valid:
            return [], -1, [], f"❌ Parameter Error: {params_msg}", "", "Ready for next generation"
        
        # Progress update
        progress_msg = "🔄 Initializing generation..."
        
        # Determine final LoRA path
        if custom_lora_path.strip():
            final_lora_path = custom_lora_path.strip()
        else:
            _, models = get_filtered_lora_choices()
            final_lora_path = get_lora_path_from_choice(lora_dropdown, models)
        
        # Start generation
        start_time = time.time()
        progress_msg = "🎨 Creating your masterpiece..."
        
        images, used_seed, file_paths = generate_image(
            prompt, negative_prompt, width, height, steps, guidance_scale, seed,
            final_lora_path, lora_scale, num_images, add_watermark_flag
        )
        
        generation_time = time.time() - start_time
        time_msg = f"{generation_time:.2f}s"
        
        # Update session stats
        stats = usage_tracker.get_stats()
        stats_html = f"""
        <div class="stats-container">
            <h4>📊 This Session</h4>
            <p><strong>Images Generated:</strong> {stats['images_generated']}</p>
            <p><strong>Average Time:</strong> {stats['avg_time_per_image']:.2f}s per image</p>
            <p><strong>Images/Minute:</strong> {stats['images_per_minute']:.1f}</p>
            <p><strong>Session Duration:</strong> {stats['session_duration']:.1f}s</p>
        </div>
        """
        
        success_msg = f"✅ Generated {len(images)} image(s) successfully!"
        
        return images, used_seed, file_paths, success_msg, time_msg, stats_html
        
    except FluxStudioError as e:
        error_msg = f"❌ Studio Error: {str(e)}"
        logger.error(error_msg)
        return [], -1, [], error_msg, "", "Error occurred - Ready for retry"
    
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], -1, [], error_msg, "", "Error occurred - Ready for retry"

def generate_with_lora_handling(
    prompt, negative_prompt, width, height, steps, guidance_scale, seed,
    lora_dropdown, custom_lora_path, lora_scale, num_images
):
    """Wrapper function to handle LoRA selection logic"""
    # Determine final LoRA path
    if custom_lora_path.strip():
        # Use custom path if provided
        final_lora_path = custom_lora_path.strip()
    else:
        # Use dropdown selection
        _, models = get_filtered_lora_choices()
        final_lora_path = get_lora_path_from_choice(lora_dropdown, models)
    
    return generate_image(
        prompt, negative_prompt, width, height, steps, guidance_scale, seed,
        final_lora_path, lora_scale, num_images
    )

def batch_generate_with_lora_handling(
    file_upload, width, height, steps, guidance_scale,
    lora_dropdown, custom_lora_path, lora_scale, output_dir
):
    """Wrapper function for batch generation with LoRA handling - Real-time updates"""
    negative_prompt = None  # FLUX doesn't use negative prompts effectively
    try:
        # Determine final LoRA path
        if custom_lora_path.strip():
            final_lora_path = custom_lora_path.strip()
        else:
            _, models = get_filtered_lora_choices()
            final_lora_path = get_lora_path_from_choice(lora_dropdown, models)
        
        # Call the actual batch generation function and yield real-time results
        for result in generate_images_from_file(
            file_upload, negative_prompt, width, height, steps, guidance_scale,
            final_lora_path, lora_scale, output_dir, create_zip=True
        ):
            yield result
            
    except Exception as e:
        yield [], f"Error in batch generation: {str(e)}", [], [], None

# Gradio UI
# Enhanced UI with custom CSS - Improved readability
custom_css = """
/* Main application styling */
.gradio-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    color: #ffffff !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.main-header-compact {
    text-align: center;
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.pro-badge {
    background: #ff6b6b;
    color: white !important;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    display: inline-block;
}

/* Stats and information containers */
.stats-container {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    color: #ffffff !important;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #34495e;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.stats-container-compact {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    color: #ffffff !important;
    padding: 8px 15px;
    border-radius: 6px;
    border: 1px solid #34495e;
    margin: 5px 0;
    text-align: center;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.stats-container h4 {
    color: #74b9ff !important;
    margin-bottom: 10px;
}

.stats-container p {
    color: #ddd !important;
    margin: 5px 0;
}

.stats-container strong {
    color: #ffffff !important;
}

/* Error and success messages */
.error-message {
    background: linear-gradient(135deg, #d63031 0%, #e17055 100%) !important;
    color: #ffffff !important;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #d63031;
    font-weight: 500;
}

.success-message {
    background: linear-gradient(135deg, #00b894 0%, #00cec9 100%) !important;
    color: #ffffff !important;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #00b894;
    font-weight: 500;
}

/* Tab styling */
.tab-nav {
    background: rgba(45, 55, 72, 0.8) !important;
}

.tab-nav .svelte-1ks3qsk {
    color: #ffffff !important;
    background: transparent !important;
}

.tab-nav .selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #ffffff !important;
}

/* Input fields and controls */
.gr-textbox, .gr-dropdown, .gr-slider {
    background: rgba(45, 55, 72, 0.9) !important;
    color: #ffffff !important;
    border: 1px solid #4a5568 !important;
    border-radius: 6px !important;
}

.gr-textbox input, .gr-dropdown select {
    background: rgba(45, 55, 72, 0.9) !important;
    color: #ffffff !important;
}

.gr-textbox label, .gr-dropdown label, .gr-slider label {
    color: #ffffff !important;
    font-weight: 500;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500;
    transition: all 0.3s ease;
}

.gr-button:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6c4ba6 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.gr-button.secondary {
    background: linear-gradient(135deg, #636e72 0%, #2d3436 100%) !important;
}

.gr-button.secondary:hover {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%) !important;
}

/* Accordion styling */
.gr-accordion {
    background: rgba(45, 55, 72, 0.8) !important;
    border: 1px solid #4a5568 !important;
    border-radius: 8px !important;
}

.gr-accordion summary {
    background: rgba(102, 126, 234, 0.8) !important;
    color: #ffffff !important;
    padding: 12px;
    border-radius: 6px;
    font-weight: 500;
}

/* Gallery styling */
.gr-gallery {
    background: rgba(45, 55, 72, 0.8) !important;
    border: 1px solid #4a5568 !important;
    border-radius: 8px !important;
}

/* File upload styling */
.gr-file {
    background: rgba(45, 55, 72, 0.8) !important;
    border: 2px dashed #4a5568 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* Markdown content styling */
.markdown-content {
    color: #ffffff !important;
}

.markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4 {
    color: #74b9ff !important;
}

.markdown-content p {
    color: #ddd !important;
}

.markdown-content strong {
    color: #ffffff !important;
}

.markdown-content code {
    background: rgba(45, 55, 72, 0.8) !important;
    color: #74b9ff !important;
    padding: 2px 4px;
    border-radius: 3px;
}

.markdown-content pre {
    background: rgba(45, 55, 72, 0.9) !important;
    color: #ffffff !important;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #4a5568;
}

/* Special styling for professional sections */
.professional-section {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #34495e;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.professional-section h3 {
    color: #74b9ff !important;
    margin-bottom: 15px;
    border-bottom: 2px solid #74b9ff;
    padding-bottom: 5px;
}

.professional-section p, .professional-section li {
    color: #ddd !important;
    line-height: 1.6;
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%) !important;
    color: #ffffff !important;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #ffffff;
}

.warning-box {
    background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%) !important;
    color: #2d3436 !important;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #2d3436;
    font-weight: 500;
}

/* Footer styling */
.footer-section {
    background: rgba(45, 55, 72, 0.9) !important;
    color: #ddd !important;
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 2px solid #4a5568;
    border-radius: 8px;
}

.footer-section strong {
    color: #74b9ff !important;
}
"""

# System status function with improved styling
def get_system_status():
    """Get current system status for display - Compact version"""
    try:
        stats = get_system_stats()
        usage_stats = usage_tracker.get_stats()
        
        status_html = f"""
        <div class="stats-container-compact">
            <span style="display: inline-block; margin-right: 20px;"><strong>📊 ID:</strong> {usage_stats['session_id']}</span>
            <span style="display: inline-block; margin-right: 20px;"><strong>🖼️ Images:</strong> {usage_stats['images_generated']}</span>
            <span style="display: inline-block; margin-right: 20px;"><strong>💾 Memory:</strong> {stats.get('memory_percent', 'N/A'):.0f}%</span>
            <span style="display: inline-block; margin-right: 20px;"><strong>🎮 GPU:</strong> {stats.get('gpu_memory_percent', 'N/A'):.0f}%</span>
        </div>
        """
        return status_html
    except Exception as e:
        return f"<div class='error-message'>Status unavailable</div>"

# Initialize LoRA data
initial_choices, initial_lora_models = get_filtered_lora_choices()

with gr.Blocks(title=f"{APP_NAME} v{APP_VERSION}", css=custom_css, theme=gr.themes.Monochrome()) as demo:
    # Professional header - Compact
    with gr.Row():
        gr.HTML(f"""
        <div class="main-header-compact">
            <h2 style="margin: 0; padding: 0;">🎨 {APP_NAME} <span class="pro-badge">v{APP_VERSION}</span></h2>
            <p style="margin: 5px 0 0 0; font-size: 14px;"><em>{LICENSE_TYPE} • Enterprise-Grade Quality</em></p>
        </div>
        """)
    
    # System status display
    with gr.Row():
        system_status = gr.HTML(get_system_status(), every=30)  # Update every 30 seconds
    
    with gr.Tab("🎨 Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls with validation
                prompt = gr.Textbox(
                    label="🎯 Creative Prompt",
                    placeholder="Describe your vision in detail... (3-2000 characters)",
                    lines=4,
                    info="Be specific about style, mood, lighting, and composition"
                )
                
                # Generation controls - Moved here for easy access
                with gr.Row():
                    generate_btn = gr.Button(
                        "✨ Generate Professional Images", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                # Professional settings section
                gr.Markdown("### ⚙️ Professional Settings")
                    
                with gr.Accordion("🔧 Advanced Quality Settings", open=False):
                    with gr.Row():
                        width = gr.Slider(
                            label="Width", 
                            minimum=256, 
                            maximum=3048, 
                            value=512, 
                            step=64,
                            info="Must be divisible by 64"
                        )
                        height = gr.Slider(
                            label="Height", 
                            minimum=256, 
                            maximum=3048, 
                            value=512, 
                            step=64,
                            info="Must be divisible by 64"
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            label="Quality Steps", 
                            minimum=10, 
                            maximum=250, 
                            value=45, 
                            step=1,
                            info="Higher = better quality, slower generation"
                        )
                        guidance_scale = gr.Slider(
                            label="Prompt Adherence", 
                            minimum=1, 
                            maximum=8, 
                            value=3.5, 
                            step=0.1,
                            info="Higher = stricter prompt following"
                        )
                    
                    with gr.Row():
                        seed = gr.Number(
                            label="🎲 Seed (-1 for random)", 
                            value=-1, 
                            precision=0,
                            info="Use same seed to reproduce results"
                        )
                        num_images = gr.Slider(
                            label="📸 Number of Images", 
                            minimum=1, 
                            maximum=4, 
                            value=1, 
                            step=1
                        )
                
                # Enhanced LoRA settings
                gr.Markdown("### 🎭 Style Enhancement (LoRA)")
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        label="🎨 Style Model",
                        choices=initial_choices,
                        value="None (No LoRA)",
                        info="Select trained style adaptations"
                    )
                    refresh_lora_btn = gr.Button("🔄", size="sm", scale=0, variant="secondary")
                
                with gr.Row():
                    lora_path = gr.Textbox(
                        label="🔗 Custom Model Path", 
                        placeholder="HuggingFace model ID or local path (e.g., username/model-name)",
                        value="",
                        info="Advanced: Direct model specification"
                    )
                
                with gr.Row():
                    lora_scale = gr.Slider(
                        label="🎛️ Style Strength", 
                        minimum=0, 
                        maximum=3.0, 
                        value=0.8, 
                        step=0.05,
                        info="Higher values = stronger style influence"
                    )
                
                # Professional LoRA Management
                with gr.Accordion("📂 Model Management", open=False):
                    gr.Markdown("**Organize your style models for better workflow**")
                    with gr.Row():
                        manage_lora_dropdown = gr.Dropdown(
                            label="Available Models",
                            choices=[name for name in scan_lora_models().keys()],
                            value=None
                        )
                    with gr.Row():
                        hide_lora_btn = gr.Button("👁️‍🗨️ Hide", size="sm", variant="secondary")
                        show_lora_btn = gr.Button("👁️ Show", size="sm", variant="secondary") 
                    lora_status = gr.Textbox(label="📊 Management Status", interactive=False, lines=2)
                
                # Progress and validation display
                with gr.Row():
                    generation_progress = gr.Textbox(
                        label="⏳ Generation Progress",
                        value="Ready to generate...",
                        interactive=False
                    )
                
            with gr.Column(scale=1):
                # Professional output section
                gr.Markdown("### 🎨 Generated Masterpieces")
                output_images = gr.Gallery(
                    label="Your Generated Images", 
                    show_label=True, 
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    preview=True
                )
                
                # Generation metadata
                with gr.Row():
                    used_seed = gr.Number(
                        label="🎲 Seed Used", 
                        visible=True,
                        interactive=False
                    )
                    generation_time = gr.Textbox(
                        label="⏱️ Generation Time",
                        interactive=False
                    )
                
                # Professional download options
                gr.Markdown("### 📥 Professional Downloads")
                download_files = gr.Files(
                    label="💾 High-Quality Images",
                    interactive=False
                )
                
                # Real-time generation stats
                with gr.Accordion("📊 Session Analytics", open=False):
                    session_stats = gr.HTML(
                        value="<p>No generations yet in this session</p>",
                        label="Session Statistics"
                    )
    
    with gr.Tab("📦 Batch Studio"):
        with gr.Row():
            with gr.Column(scale=1):
                # Enhanced file upload options
                gr.Markdown("### 📂 Import Creative Prompts")
                with gr.Tabs():
                    with gr.Tab("📁 Upload File"):
                        batch_file_upload = gr.File(
                            label="🎯 Upload Creative Brief (JSON/TXT)",
                            file_types=[".json", ".txt"],
                            type="filepath"
                        )
                        gr.Markdown("""
                        **Supported formats:**
                        - **JSON**: `[{"prompt": "..."}, "simple prompt", ...]`
                        - **TXT**: One prompt per line
                        """)
                    
                    with gr.Tab("📝 Manual Path"):
                        json_file_path = gr.Textbox(
                            label="File Path",
                            placeholder="Path to your JSON/TXT file with prompts",
                            value="/workspace/Lora_Trainer_Imgen_Flux/example_prompts.json"
                        )
                
                # Generate buttons - Moved here for easy access
                with gr.Row():
                    batch_generate_btn = gr.Button("🚀 Generate Batch", variant="primary", scale=2)
                    batch_generate_manual_btn = gr.Button("📝 Generate from Path", variant="secondary", scale=1)
                
                # Batch settings
                gr.Markdown("### Batch Settings")
                
                # LoRA settings for batch
                gr.Markdown("### Batch LoRA Settings")
                with gr.Row():
                    batch_lora_dropdown = gr.Dropdown(
                        label="Select LoRA Model for Batch",
                        choices=initial_choices,
                        value="None (No LoRA)",
                        info="Choose from your trained LoRA models"
                    )
                    batch_refresh_lora_btn = gr.Button("🔄", size="sm", scale=0)
                
                batch_lora_path = gr.Textbox(
                    label="Custom LoRA Path for Batch",
                    placeholder="Or enter custom path/HF model (e.g., username/model-name)",
                    value="",
                    info="Leave empty to use selected LoRA above"
                )
                
                batch_lora_scale = gr.Slider(
                    label="LoRA Scale for Batch", 
                    minimum=0, 
                    maximum=3.0, 
                    value=0.8, 
                    step=0.05
                )
                
                output_directory = gr.Textbox(
                    label="Output Directory",
                    placeholder="Directory to save generated images",
                    value="/workspace/Lora_Trainer_Imgen_Flux/batch_output"
                )
                
                with gr.Row():
                    batch_width = gr.Slider(label="Width", minimum=384, maximum=2700, value=512, step=64)
                    batch_height = gr.Slider(label="Height", minimum=384, maximum=2700, value=512, step=64)
                
                with gr.Row():
                    batch_steps = gr.Slider(label="Steps", minimum=15, maximum=300, value=45, step=1)
                    batch_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=8, value=3.5, step=0.1)
                
            with gr.Column(scale=1):
                # Output gallery for batch
                gr.Markdown("### 🖼️ Generated Images")
                batch_output_images = gr.Gallery(
                    label="Batch Generated Images", 
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                batch_status = gr.Textbox(
                    label="📊 Batch Status", 
                    value="Ready to process batch",
                    interactive=False
                )
                
                # Download options
                gr.Markdown("### 📥 Download Options")
                batch_download_files = gr.Files(
                    label="Individual Images",
                    interactive=False
                )
                
                batch_zip_download = gr.File(
                    label="📦 Complete Batch ZIP",
                    interactive=False
                )
                
                batch_seeds = gr.Textbox(
                    label="🎲 Seeds Used", 
                    value="",
                    interactive=False,
                    lines=3
                )
                
                with gr.Row():
                    batch_clear_btn = gr.Button("🗑️ Clear Results", variant="secondary")
    
    with gr.Tab("📚 Professional Guide"):
        # Professional documentation
        gr.HTML("""
        <div class="professional-section" style="text-align: center;">
            <h2>📚 Professional User Guide</h2>
            <p>Master the art of AI image generation with industry best practices</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎯 Professional Prompting"):
                gr.Markdown("""
                ## 🎨 Advanced Prompt Engineering
                
                ### ✨ Professional Prompting Techniques
                * **Be Specific**: Instead of "a car", use "a sleek red Ferrari 488 GTB on a mountain road at sunset"
                * **Style Direction**: Add style qualifiers like "photorealistic", "oil painting", "digital art", "concept art"
                * **Lighting**: Specify lighting conditions: "golden hour", "dramatic lighting", "soft studio lighting"
                * **Composition**: Include framing: "close-up portrait", "wide angle", "aerial view", "macro photography"
                * **Mood & Atmosphere**: Set the emotional tone: "serene", "dramatic", "mysterious", "energetic"
                
                ### � Technical Parameters
                * **Steps**: 20-30 for quick tests, 50+ for final quality, 100+ for maximum detail
                * **Guidance Scale**: 3-7 for creative freedom, 7-12 for strict prompt adherence
                * **Dimensions**: Use multiples of 64 for optimal results, standard sizes: 512x512, 768x768, 1024x1024
                
                ### � FLUX-Specific Tips
                * **No Negative Prompts**: FLUX works best with positive descriptions only
                * **Quality Control**: Build quality requirements into your positive prompt
                * **Style Specification**: Be explicit about desired artistic style and quality level
                """)
                
            with gr.Tab("🎭 Style Enhancement"):
                gr.Markdown("""
                ## 🎨 LoRA Style System
                
                ### 🔧 Professional LoRA Usage
                * **Style LoRAs**: Transform your images with artistic styles (watercolor, oil painting, anime, etc.)
                * **Character LoRAs**: Generate consistent characters across multiple images
                * **Concept LoRAs**: Add specific objects, clothing, or scenarios
                * **Quality LoRAs**: Enhance detail, lighting, or technical aspects
                
                ### ⚡ Scale Guidelines
                * **0.3-0.5**: Subtle influence, maintains base model strength
                * **0.6-0.8**: Balanced application (recommended for most uses)
                * **0.9-1.2**: Strong style influence
                * **1.3+**: Maximum style dominance (may cause artifacts)
                
                ### 🔗 HuggingFace Integration
                * Use format: `username/model-name` for public models
                * Private models: Use access tokens in .env file
                * Local models: Full file path to .safetensors file
                """)
                
            with gr.Tab("🏭 Batch Production"):
                gr.Markdown("""
                ## 📦 Enterprise Batch Processing
                
                ### 📂 Input File Formats
                
                **JSON Format (Advanced)**:
                ```json
                [
                  {
                    "prompt": "A majestic lion in African savanna",
                    "style": "photorealistic wildlife photography",
                    "metadata": {"client": "Nature Magazine"}
                  },
                  {
                    "prompt": "Modern architecture building",
                    "style": "architectural visualization"
                  },
                  "Simple prompt string also works"
                ]
                ```
                
                **Text Format (Simple)**:
                ```
                A majestic lion in African savanna, photorealistic
                Modern architecture building, clean lines
                # Comments start with # and are ignored
                Portrait of a professional woman, corporate headshot
                ```
                
                ### 🎯 Production Workflow
                1. **Prepare Prompts**: Create comprehensive prompt list
                2. **Test Settings**: Run single images first to perfect parameters
                3. **Batch Process**: Upload file and start batch generation
                4. **Quality Review**: Check outputs in gallery preview
                5. **Download**: Use ZIP download for complete project delivery
                
                ### 📊 Output Organization
                * **Timestamped Folders**: Each batch gets unique timestamp
                * **Descriptive Names**: Files include prompt preview and seed
                * **Metadata Tracking**: JSON file with all generation settings
                * **Professional ZIP**: Ready for client delivery
                """)
                
            with gr.Tab("⚙️ Technical Specs"):
                gr.Markdown(f"""
                ## 🔧 System Specifications
                
                ### 💻 Software Information
                * **Application**: {APP_NAME}
                * **Version**: {APP_VERSION}
                * **License**: {LICENSE_TYPE}
                * **Engine**: FLUX.1-dev Professional
                * **Precision**: bfloat16 (optimal quality/speed)
                
                ### 🎛️ Generation Limits
                * **Image Dimensions**: 256x256 to 2048x2048 pixels
                * **Quality Steps**: 10-150 (professional range: 25-50)
                * **Guidance Scale**: 1.0-20.0 (optimal: 3.0-7.0)
                * **Batch Size**: Up to 1000 prompts per batch
                * **Concurrent Images**: 1-4 per generation
                
                ### 🚀 Performance Features
                * **Memory Optimization**: Automatic CUDA memory management
                * **Progress Tracking**: Real-time generation updates
                * **Error Recovery**: Automatic fallback for failed generations
                * **Watermarking**: Professional branding system
                * **Usage Analytics**: Session tracking and statistics
                
                ### 📁 File Management
                * **Auto-cleanup**: Intelligent cache management
                * **Format Support**: PNG (high quality), JPEG (smaller size)
                * **Metadata Embedding**: EXIF data with generation parameters
                * **Batch Organization**: Automated folder structure
                """)
                
            with gr.Tab("🎓 Best Practices"):
                gr.Markdown("""
                ## 🏆 Industry Best Practices
                
                ### 💡 Creative Workflow
                1. **Research Phase**: Study reference images and styles
                2. **Prompt Development**: Write detailed, specific descriptions
                3. **Parameter Testing**: Find optimal settings for your style
                4. **Iteration**: Refine prompts based on results
                5. **Final Production**: Use proven settings for client work
                
                ### 📈 Quality Optimization
                * **Consistency**: Use same seed for variations
                * **A/B Testing**: Compare different parameter settings
                * **Style Guides**: Document successful prompt patterns
                * **Client Feedback**: Iterate based on professional requirements
                
                ### 🔒 Professional Standards
                * **Copyright Compliance**: Avoid reproducing copyrighted materials
                * **Client Communication**: Share generation parameters for revisions
                * **File Organization**: Maintain professional folder structures
                * **Backup Strategy**: Keep original prompts and high-quality outputs
                
                ### ⚡ Efficiency Tips
                * **Batch Similar**: Group similar prompts for efficient processing
                * **Template Prompts**: Create reusable prompt templates
                * **Style Libraries**: Build collections of effective LoRA combinations
                * **Quality Control**: Implement systematic review processes
                """)
        
        # System information section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; border: 1px solid #dee2e6;">
            <h3>🛠️ Technical Support</h3>
            <p><strong>For technical support and licensing inquiries:</strong></p>
            <p>• Professional licensing and custom deployment options available</p>
            <p>• Enterprise support with SLA guarantees</p>
            <p>• Custom model training and integration services</p>
            <p>• API access for automated workflows</p>
        </div>
        """)
    
    with gr.Tab("📊 Analytics"):
        # Real-time analytics dashboard
        gr.HTML("""
        <div class="professional-section" style="text-align: center;">
            <h2>📊 Professional Analytics Dashboard</h2>
            <p>Monitor your creative productivity and system performance</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Session Performance")
                session_analytics = gr.HTML(
                    value=get_system_status(),
                    every=10  # Update every 10 seconds
                )
                
                gr.Markdown("### 🖥️ System Resources")
                system_resources = gr.HTML(every=15)
                
            with gr.Column():
                gr.Markdown("### 🎨 Generation History")
                generation_log = gr.Textbox(
                    label="Recent Generations",
                    lines=10,
                    interactive=False,
                    value="No generations yet in this session"
                )
                
                refresh_analytics_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")
    
    # Add footer with professional branding
    gr.HTML(f"""
    <div class="footer-section">
        <p><strong>{APP_NAME} v{APP_VERSION}</strong> • {LICENSE_TYPE}</p>
        <p>Professional AI Image Generation Platform • Enterprise-Grade Quality</p>
        <p><em>Powered by FLUX.1-dev • Optimized for Commercial Use</em></p>
    </div>
    """)

    # Set up event handlers
    def clear_outputs():
        return [None, None, None, "Ready for next generation", "", "No generations yet in this session"]
    
    def clear_batch_outputs():
        return [None, "Ready to process batch", None, "", None]
    
    # Legacy function for manual path input
    def generate_from_manual_path(
        json_file_path,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        lora_path,
        lora_scale,
        output_dir
    ):
        """Generate images from manual file path (legacy support) - Real-time updates"""
        negative_prompt = None  # FLUX doesn't use negative prompts effectively
        try:
            # Create a temporary file object for the manual path
            class FileWrapper:
                def __init__(self, path):
                    self.name = path
            
            file_wrapper = FileWrapper(json_file_path)
            
            # Call the generator function and yield real-time results
            for result in generate_images_from_file(
                file_wrapper,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                lora_path,
                lora_scale,
                output_dir,
                create_zip=True
            ):
                yield result
                
        except Exception as e:
            yield [], f"Error: {str(e)}", [], [], None
    
    # Enhanced single image generation
    generate_btn.click(
        fn=generate_with_enhanced_features,
        inputs=[
            prompt, 
            width, 
            height, 
            steps, 
            guidance_scale, 
            seed,
            lora_dropdown,
            lora_path,
            lora_scale,
            num_images
        ],
        outputs=[output_images, used_seed, download_files, generation_progress, generation_time, session_stats]
    )
    
    # Batch generation from uploaded file
    batch_generate_btn.click(
        fn=batch_generate_with_lora_handling,
        inputs=[
            batch_file_upload,
            batch_width,
            batch_height,
            batch_steps,
            batch_guidance_scale,
            batch_lora_dropdown,
            batch_lora_path,
            batch_lora_scale,
            output_directory
        ],
        outputs=[batch_output_images, batch_status, batch_download_files, batch_seeds, batch_zip_download]
    )
    
    # Batch generation from manual path
    batch_generate_manual_btn.click(
        fn=generate_from_manual_path,
        inputs=[
            json_file_path,
            batch_width,
            batch_height,
            batch_steps,
            batch_guidance_scale,
            batch_lora_path,
            batch_lora_scale,
            output_directory
        ],
        outputs=[batch_output_images, batch_status, batch_download_files, batch_seeds, batch_zip_download]
    )
    
    # Clear buttons
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[output_images, used_seed, download_files]
    )
    
    batch_clear_btn.click(
        fn=clear_batch_outputs,
        inputs=[],
        outputs=[batch_output_images, batch_status, batch_download_files, batch_seeds, batch_zip_download]
    )
    
    # LoRA Management Event Handlers  
    refresh_lora_btn.click(
        fn=refresh_lora_list,
        inputs=[],
        outputs=[lora_dropdown, manage_lora_dropdown]
    )
    
    batch_refresh_lora_btn.click(
        fn=refresh_lora_list,
        inputs=[],
        outputs=[batch_lora_dropdown, manage_lora_dropdown]
    )
    
    lora_dropdown.change(
        fn=handle_lora_selection,
        inputs=[lora_dropdown],
        outputs=[lora_path]
    )
    
    batch_lora_dropdown.change(
        fn=handle_lora_selection,
        inputs=[batch_lora_dropdown],
        outputs=[batch_lora_path]
    )
    
    hide_lora_btn.click(
        fn=lambda x: manage_lora_visibility(x, "hide"),
        inputs=[manage_lora_dropdown],
        outputs=[lora_dropdown, manage_lora_dropdown, lora_status]
    )
    
    show_lora_btn.click(
        fn=lambda x: manage_lora_visibility(x, "show"),
        inputs=[manage_lora_dropdown], 
        outputs=[lora_dropdown, manage_lora_dropdown, lora_status]
    )

# Launch the app
demo.launch(share=True, quiet=True, inbrowser=True)