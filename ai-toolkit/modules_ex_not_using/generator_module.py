"""
FLUX Image Generator Module
===========================

Modularized version of the FLUX Image Generator for the Unified AI Toolkit.
This module handles image generation with LoRA support and advanced features.
"""

import os
import sys
import gc
import json
import time
import random
import tempfile
import zipfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import gradio as gr
import torch
from PIL import Image
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

# Add current directory to path
current_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(current_dir))

# Set up optimal cache location
CACHE_DIR = os.path.expanduser("/root/.caches")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure environment variables for better caching
os.environ.update({
    "HF_HOME": CACHE_DIR,
    "TRANSFORMERS_CACHE": CACHE_DIR,
    "HUGGINGFACE_HUB_CACHE": CACHE_DIR,
    "SAFETENSORS_CACHE_DIR": CACHE_DIR,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
})

# Import FLUX pipeline components
try:
    from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
    FLUX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ FLUX components not available: {e}")
    FLUX_AVAILABLE = False

class FluxGenerator:
    """FLUX image generation pipeline with LoRA support"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self._initialized = False
    
    def initialize_pipeline(self):
        """Initialize the FLUX pipeline"""
        if self._initialized or not FLUX_AVAILABLE:
            return
        
        print("🔄 Loading FLUX pipeline, please wait...")
        
        try:
            # Free up CUDA memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load required models
            print("📝 Loading CLIP text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14-336", 
                torch_dtype=self.dtype, 
                cache_dir=CACHE_DIR
            ).to(self.device)
            
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14-336", 
                cache_dir=CACHE_DIR
            )
            
            print("🎨 Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                subfolder="vae", 
                torch_dtype=self.dtype, 
                cache_dir=CACHE_DIR
            ).to(self.device)
            
            print("🧠 Loading T5 text encoder...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                "google/t5-v1_1-xxl", 
                torch_dtype=self.dtype, 
                cache_dir=CACHE_DIR
            ).to(self.device)
            
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "google/t5-v1_1-xxl", 
                legacy=True, 
                cache_dir=CACHE_DIR
            )
            
            scheduler = FlowMatchEulerDiscreteScheduler()
            
            print("🔄 Assembling pipeline...")
            # Initialize the pipeline
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                torch_dtype=self.dtype,
                scheduler=scheduler,
                cache_dir=CACHE_DIR
            )
            
            # Move to GPU
            self.pipe.to(self.device)
            
            # Optimize memory usage
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ Using xformers for memory efficient attention")
            except Exception:
                print("⚠️ xformers not available. Using default attention mechanism.")
            
            self._initialized = True
            print("✅ FLUX pipeline loaded and ready")
            
        except Exception as e:
            print(f"❌ Failed to initialize FLUX pipeline: {e}")
            self.pipe = None
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = -1,
        lora_path: str = "",
        lora_scale: float = 0.8,
        num_images: int = 1
    ) -> Tuple[List[Image.Image], int]:
        """Generate images using FLUX pipeline"""
        
        if not self._initialized:
            self.initialize_pipeline()
        
        if not self.pipe:
            raise Exception("FLUX pipeline not available")
        
        # Unload any previous LoRA weights
        if hasattr(self.pipe, 'unload_lora_weights'):
            try:
                self.pipe.unload_lora_weights()
            except:
                pass
        
        # Load LoRA weights if specified
        if lora_path and lora_path.strip() and os.path.exists(lora_path.strip()):
            try:
                self.pipe.load_lora_weights(lora_path.strip())
                print(f"✅ Loaded LoRA: {lora_path}")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA: {e}")
        
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        # Configure optimal parameters
        safe_width = (width // 16) * 16  # Ensure width is divisible by 16
        safe_height = (height // 16) * 16  # Ensure height is divisible by 16
        safe_guidance = min(guidance_scale, 12.0)
        
        print(f"🔄 Generating image with seed: {seed}")
        
        try:
            # Generate images
            images = self.pipe(
                prompt=prompt,
                prompt_2=prompt, 
                negative_prompt=negative_prompt if negative_prompt else None,
                height=safe_height,
                width=safe_width,
                guidance_scale=safe_guidance,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images_per_prompt=num_images,
                joint_attention_kwargs={"scale": lora_scale} if lora_path.strip() else None,
                output_type="pil"
            ).images
            
            if images:
                print(f"✅ Generated {len(images)} image(s) successfully")
                return images, seed
            else:
                raise Exception("No images generated")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            # Try fallback with safer settings
            try:
                print("🔄 Trying fallback generation...")
                images = self.pipe(
                    prompt=prompt,
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=10,
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"
                ).images
                return images, seed
            except Exception as fallback_e:
                raise Exception(f"Generation failed: {e}, Fallback also failed: {fallback_e}")
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        if self.pipe:
            try:
                del self.pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self._initialized = False
                print("🧹 FLUX pipeline cleaned up")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

class LoRAManager:
    """Manages LoRA models and provides selection interface"""
    
    def __init__(self, lora_output_dir: str = None):
        self.lora_output_dir = lora_output_dir or f"{current_dir}/output"
        self.filter_file = f"{current_dir}/lora_filter.json"
    
    def scan_lora_models(self) -> Dict[str, Dict]:
        """Scan the LoRA output directory for available models"""
        lora_models = {}
        
        if not os.path.exists(self.lora_output_dir):
            return lora_models
        
        try:
            for item in os.listdir(self.lora_output_dir):
                item_path = os.path.join(self.lora_output_dir, item)
                if os.path.isdir(item_path):
                    # Look for LoRA files
                    for file in os.listdir(item_path):
                        if file.endswith('.safetensors') and 'lora' in file.lower():
                            file_path = os.path.join(item_path, file)
                            file_size = os.path.getsize(file_path) // (1024 * 1024)  # MB
                            
                            lora_models[item] = {
                                'path': file_path,
                                'size': f"{file_size}MB",
                                'file': file,
                                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
                            }
                            break
        except Exception as e:
            print(f"Error scanning LoRA models: {e}")
        
        return lora_models
    
    def get_lora_choices(self) -> List[str]:
        """Get LoRA choices for dropdown"""
        lora_models = self.scan_lora_models()
        choices = ["None (No LoRA)"]
        
        for name, info in lora_models.items():
            choice = f"{name} ({info['size']})"
            choices.append(choice)
        
        return choices
    
    def get_lora_path_from_choice(self, choice: str) -> str:
        """Convert dropdown choice back to actual LoRA path"""
        if choice == "None (No LoRA)" or not choice:
            return ""
        
        # Extract the model name from the choice (before the size info)
        model_name = choice.split(" (")[0]
        lora_models = self.scan_lora_models()
        
        if model_name in lora_models:
            return lora_models[model_name]['path']
        
        return ""

def parse_prompt_file(uploaded_file) -> List[str]:
    """Parse uploaded JSON or text file to extract prompts"""
    try:
        file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as JSON first
        if file_path.lower().endswith('.json'):
            try:
                data = json.loads(content)
                prompts = []
                
                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            prompts.append(item)
                        elif isinstance(item, dict):
                            # Look for common prompt keys
                            for key in ['prompt', 'text', 'caption', 'description']:
                                if key in item:
                                    prompts.append(item[key])
                                    break
                elif isinstance(data, dict):
                    # Look for prompts in various structures
                    if 'prompts' in data:
                        prompts = data['prompts']
                    elif 'captions' in data:
                        captions = data['captions']
                        if isinstance(captions, dict):
                            prompts = list(captions.values())
                        else:
                            prompts = captions
                    else:
                        # Try to extract all string values
                        for value in data.values():
                            if isinstance(value, str):
                                prompts.append(value)
                
                return prompts
            except json.JSONDecodeError:
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

def create_batch_zip(output_dir: str, batch_name: str = None) -> str:
    """Create a ZIP file containing all images from batch processing"""
    try:
        if not os.path.exists(output_dir):
            return None
        
        if batch_name is None:
            batch_name = f"batch_{int(time.time())}"
        
        # Create ZIP in parent directory
        parent_dir = os.path.dirname(output_dir)
        zip_path = os.path.join(parent_dir, f"{batch_name}.zip")
        
        print(f"📦 Creating ZIP file: {zip_path}")
        
        image_count = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        # Use relative path in ZIP
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
                        image_count += 1
        
        if image_count > 0:
            print(f"✅ Created ZIP with {image_count} images")
            return zip_path
        else:
            os.remove(zip_path)
            return None
        
    except Exception as e:
        print(f"❌ Error creating batch ZIP: {e}")
        return None

def save_images_to_directory(images: List[Image.Image], output_dir: str) -> List[str]:
    """Save images to a specific directory"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, image in enumerate(images):
        timestamp = int(time.time())
        file_path = os.path.join(output_dir, f"generated_image_{timestamp}_{i}.png")
        image.save(file_path, "PNG", quality=95)
        saved_paths.append(file_path)
    
    return saved_paths

def create_generator_interface() -> gr.Blocks:
    """Create the FLUX Image Generator interface"""
    
    # Initialize components
    flux_generator = None
    lora_manager = LoRAManager()
    
    def initialize_generator():
        """Initialize the FLUX generator"""
        nonlocal flux_generator
        if flux_generator is None:
            flux_generator = FluxGenerator()
        return flux_generator
    
    def generate_single_image(
        prompt, negative_prompt, width, height, steps, guidance_scale, seed,
        lora_dropdown, custom_lora_path, lora_scale, num_images
    ):
        """Generate single image with LoRA handling"""
        try:
            generator = initialize_generator()
            
            # Determine final LoRA path
            if custom_lora_path.strip():
                final_lora_path = custom_lora_path.strip()
            else:
                final_lora_path = lora_manager.get_lora_path_from_choice(lora_dropdown)
            
            images, used_seed = generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                lora_path=final_lora_path,
                lora_scale=lora_scale,
                num_images=num_images
            )
            
            # Save images temporarily for download
            temp_dir = tempfile.mkdtemp(prefix="flux_generation_")
            saved_paths = save_images_to_directory(images, temp_dir)
            
            return images, f"✅ Generated {len(images)} image(s) with seed: {used_seed}", saved_paths
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            return [], error_msg, []
    
    def generate_batch_images(
        file_upload, negative_prompt, width, height, steps, guidance_scale,
        lora_dropdown, custom_lora_path, lora_scale, output_dir
    ):
        """Generate batch images from uploaded file"""
        try:
            if not file_upload:
                return [], "❌ Please upload a file with prompts", [], [], None
            
            # Parse prompts from uploaded file
            prompts = parse_prompt_file(file_upload)
            
            if not prompts:
                return [], "❌ No valid prompts found in uploaded file", [], [], None
            
            generator = initialize_generator()
            
            # Determine final LoRA path
            if custom_lora_path.strip():
                final_lora_path = custom_lora_path.strip()
            else:
                final_lora_path = lora_manager.get_lora_path_from_choice(lora_dropdown)
            
            # Create timestamped output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_output_dir = os.path.join(output_dir, f"batch_{timestamp}")
            os.makedirs(batch_output_dir, exist_ok=True)
            
            generated_images = []
            all_image_paths = []
            seeds_used = []
            
            # Save batch metadata
            metadata = {
                "batch_id": timestamp,
                "total_prompts": len(prompts),
                "settings": {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "lora_path": final_lora_path,
                    "lora_scale": lora_scale
                },
                "prompts": [],
                "generation_time": None
            }
            
            start_time = time.time()
            
            # Process each prompt
            for i, prompt in enumerate(prompts):
                try:
                    print(f"🎨 Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
                    
                    images, seed = generator.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        seed=-1,  # Random seed for each image
                        lora_path=final_lora_path,
                        lora_scale=lora_scale,
                        num_images=1
                    )
                    
                    if images:
                        # Save image
                        image_filename = f"image_{i+1:04d}_seed_{seed}.png"
                        image_path = os.path.join(batch_output_dir, image_filename)
                        images[0].save(image_path, "PNG", quality=95)
                        
                        generated_images.extend(images)
                        all_image_paths.append(image_path)
                        seeds_used.append(seed)
                        
                        # Add to metadata
                        metadata["prompts"].append({
                            "index": i + 1,
                            "prompt": prompt,
                            "seed": seed,
                            "filename": image_filename,
                            "generated": True
                        })
                    else:
                        metadata["prompts"].append({
                            "index": i + 1,
                            "prompt": prompt,
                            "seed": None,
                            "filename": None,
                            "generated": False
                        })
                        
                except Exception as e:
                    print(f"❌ Failed to generate image {i+1}: {e}")
                    metadata["prompts"].append({
                        "index": i + 1,
                        "prompt": prompt,
                        "seed": None,
                        "filename": None,
                        "generated": False,
                        "error": str(e)
                    })
            
            # Save metadata
            metadata["generation_time"] = time.time() - start_time
            metadata_path = os.path.join(batch_output_dir, "batch_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create ZIP file
            zip_path = create_batch_zip(batch_output_dir, f"batch_{timestamp}")
            
            success_count = len([p for p in metadata["prompts"] if p["generated"]])
            status_msg = f"✅ Batch completed! Generated {success_count}/{len(prompts)} images successfully."
            
            return generated_images, status_msg, all_image_paths, seeds_used, zip_path
            
        except Exception as e:
            return [], f"❌ Batch generation failed: {str(e)}", [], [], None
    
    def refresh_lora_list():
        """Refresh the LoRA dropdown list"""
        choices = lora_manager.get_lora_choices()
        return gr.update(choices=choices)
    
    def handle_lora_selection(choice):
        """Handle LoRA dropdown selection"""
        path = lora_manager.get_lora_path_from_choice(choice)
        return path
    
    def cleanup_generator():
        """Cleanup the generator model"""
        nonlocal flux_generator
        if flux_generator:
            flux_generator.cleanup()
            flux_generator = None
        return "🧹 Generator model cleaned up successfully"
    
    # Custom CSS
    custom_css = """
    .generator-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .generate-btn {
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
                🎨 FLUX Image Generator
            </h1>
            <p style="font-size: 1.2em; color: #666;">
                Advanced AI image generation with LoRA support and batch processing
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Single Generation Tab
            with gr.Tab("🖼️ Single Generation"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="generator-container"):
                        gr.HTML('<h3 style="text-align: center;">⚙️ Generation Settings</h3>')
                        
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful landscape with mountains and a lake",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="blurry, low quality, distorted",
                            lines=2
                        )
                        
                        with gr.Row():
                            width = gr.Number(label="Width", value=1024, minimum=256, maximum=2048, step=64)
                            height = gr.Number(label="Height", value=1024, minimum=256, maximum=2048, step=64)
                        
                        with gr.Row():
                            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
                            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=3.5, step=0.5)
                        
                        with gr.Row():
                            seed = gr.Number(label="Seed (-1 for random)", value=-1, minimum=-1, maximum=2147483647)
                            num_images = gr.Number(label="Number of Images", value=1, minimum=1, maximum=4, step=1)
                        
                        # LoRA Settings
                        with gr.Accordion("🎭 LoRA Settings", open=False):
                            with gr.Row():
                                lora_dropdown = gr.Dropdown(
                                    label="Select LoRA Model",
                                    choices=lora_manager.get_lora_choices(),
                                    value="None (No LoRA)",
                                    interactive=True
                                )
                                refresh_lora_btn = gr.Button("🔄 Refresh", size="sm")
                            
                            custom_lora_path = gr.Textbox(
                                label="Custom LoRA Path (Optional)",
                                placeholder="/path/to/your/lora.safetensors",
                                info="Override dropdown selection with custom path"
                            )
                            lora_scale = gr.Slider(
                                label="LoRA Scale",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.8,
                                step=0.1,
                                info="Strength of LoRA effect"
                            )
                        
                        generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg", elem_classes="generate-btn")
                        cleanup_btn = gr.Button("🧹 Cleanup Model", variant="secondary")
                    
                    with gr.Column(scale=1, elem_classes="generator-container"):
                        gr.HTML('<h3 style="text-align: center;">🖼️ Generated Images</h3>')
                        
                        output_images = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            elem_id="output_gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                        
                        generation_status = gr.Markdown()
                        download_files = gr.File(label="Download Images", file_count="multiple", interactive=False)
            
            # Batch Generation Tab
            with gr.Tab("📦 Batch Generation"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="generator-container"):
                        gr.HTML('<h3 style="text-align: center;">📁 Batch Settings</h3>')
                        
                        batch_file_upload = gr.File(
                            label="Upload Prompts File",
                            file_types=[".txt", ".json"],
                            info="TXT file (one prompt per line) or JSON file with prompts"
                        )
                        
                        batch_negative_prompt = gr.Textbox(
                            label="Negative Prompt (Applied to all)",
                            placeholder="blurry, low quality",
                            lines=2
                        )
                        
                        with gr.Row():
                            batch_width = gr.Number(label="Width", value=1024, minimum=256, maximum=2048, step=64)
                            batch_height = gr.Number(label="Height", value=1024, minimum=256, maximum=2048, step=64)
                        
                        with gr.Row():
                            batch_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
                            batch_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=3.5, step=0.5)
                        
                        # Batch LoRA Settings
                        with gr.Accordion("🎭 LoRA Settings", open=False):
                            batch_lora_dropdown = gr.Dropdown(
                                label="Select LoRA Model",
                                choices=lora_manager.get_lora_choices(),
                                value="None (No LoRA)",
                                interactive=True
                            )
                            batch_custom_lora_path = gr.Textbox(
                                label="Custom LoRA Path (Optional)",
                                placeholder="/path/to/your/lora.safetensors"
                            )
                            batch_lora_scale = gr.Slider(
                                label="LoRA Scale",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.8,
                                step=0.1
                            )
                        
                        output_directory = gr.Textbox(
                            label="Output Directory",
                            value="./generated_images",
                            info="Directory to save batch images"
                        )
                        
                        batch_generate_btn = gr.Button("🚀 Generate Batch", variant="primary", size="lg", elem_classes="generate-btn")
                    
                    with gr.Column(scale=1, elem_classes="generator-container"):
                        gr.HTML('<h3 style="text-align: center;">📊 Batch Results</h3>')
                        
                        batch_output_images = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            columns=3,
                            rows=3,
                            height="auto"
                        )
                        
                        batch_status = gr.Markdown()
                        batch_download_files = gr.File(label="Download Individual Images", file_count="multiple", interactive=False)
                        batch_zip_download = gr.File(label="Download Batch ZIP", interactive=False)
        
        # Event handlers
        generate_btn.click(
            fn=generate_single_image,
            inputs=[
                prompt, negative_prompt, width, height, steps, guidance_scale, seed,
                lora_dropdown, custom_lora_path, lora_scale, num_images
            ],
            outputs=[output_images, generation_status, download_files]
        )
        
        batch_generate_btn.click(
            fn=generate_batch_images,
            inputs=[
                batch_file_upload, batch_negative_prompt, batch_width, batch_height,
                batch_steps, batch_guidance_scale, batch_lora_dropdown,
                batch_custom_lora_path, batch_lora_scale, output_directory
            ],
            outputs=[batch_output_images, batch_status, batch_download_files, gr.State(), batch_zip_download]
        )
        
        cleanup_btn.click(
            fn=cleanup_generator,
            outputs=[generation_status]
        )
        
        refresh_lora_btn.click(
            fn=refresh_lora_list,
            outputs=[lora_dropdown]
        )
        
        lora_dropdown.change(
            fn=handle_lora_selection,
            inputs=[lora_dropdown],
            outputs=[custom_lora_path]
        )
        
        batch_lora_dropdown.change(
            fn=handle_lora_selection,
            inputs=[batch_lora_dropdown],
            outputs=[batch_custom_lora_path]
        )
    
    return interface