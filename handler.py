"""
RunPod Serverless Handler for FLUX Image Generation
Designed for RunPod serverless GPU endpoints
"""

import os
import torch
import random
import base64
import io
import time
from PIL import Image

# Set up optimal cache location for runpod
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure environment variables for better caching on runpod
os.environ.update({
    "HF_HOME": CACHE_DIR,
    "TRANSFORMERS_CACHE": CACHE_DIR,
    "HUGGINGFACE_HUB_CACHE": CACHE_DIR,
    "SAFETENSORS_CACHE_DIR": CACHE_DIR,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
})

# Global variables for lazy loading
pipe = None
device = None
dtype = None


def load_models():
    """Load all required models - called once during cold start"""
    global pipe, device, dtype
    
    if pipe is not None:
        return  # Already loaded
    
    from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
    from huggingface_hub import login
    
    # Login to Huggingface if token is available
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️ HF_TOKEN not found. Some models may not be accessible.")
    
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
        cache_dir=CACHE_DIR
    ).to(device)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14-336", 
        cache_dir=CACHE_DIR
    )
    
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        subfolder="vae", 
        torch_dtype=dtype, 
        cache_dir=CACHE_DIR
    ).to(device)
    
    # Load T5 models for enhanced text understanding
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "google/t5-v1_1-xxl", 
        torch_dtype=dtype, 
        cache_dir=CACHE_DIR
    ).to(device)
    
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        "google/t5-v1_1-xxl", 
        legacy=True, 
        cache_dir=CACHE_DIR
    )
    
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    print("🔄 Loading the pipeline, please wait...")
    
    # Initialize the pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        torch_dtype=dtype,
        scheduler=scheduler,
        cache_dir=CACHE_DIR
    )
    
    # Move to GPU
    pipe.to(device)
    
    # Optimize memory usage
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ Using xformers for memory efficient attention")
    except Exception:
        print("⚠️ xformers not available. Using default attention mechanism.")
    
    print("✅ Model loaded and ready")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.5,
    seed: int = -1,
    lora_path: str = "",
    lora_scale: float = 0.8,
    num_images: int = 1
):
    """Generate images based on the provided parameters"""
    global pipe
    
    # Ensure models are loaded
    load_models()
    
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
    
    # Configure optimal parameters
    safe_width = (width // 16) * 16
    safe_height = (height // 16) * 16
    safe_guidance = min(guidance_scale, 12.0)
    
    full_prompt = f"{prompt}"
    
    print(f"🔄 Generating image with seed: {seed}")
    
    try:
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
            joint_attention_kwargs={"scale": lora_scale} if lora_path and lora_path.strip() else None,
            output_type="pil"
        ).images
        
        return images, seed
        
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
            
            return fallback_images, seed
        except Exception as fallback_e:
            print(f"❌ Fallback generation also failed: {fallback_e}")
            raise


def handler(job):
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "prompt": "A beautiful landscape",
            "negative_prompt": "low quality, blurry",  # optional
            "width": 512,  # optional, default 512
            "height": 512,  # optional, default 512
            "num_inference_steps": 25,  # optional, default 25
            "guidance_scale": 3.5,  # optional, default 3.5
            "seed": -1,  # optional, -1 for random
            "lora_path": "",  # optional, HuggingFace model path
            "lora_scale": 0.8,  # optional, default 0.8
            "num_images": 1  # optional, default 1
        }
    }
    """
    job_input = job.get("input", {})
    
    # Extract parameters with defaults
    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "Prompt is required"}
    
    negative_prompt = job_input.get("negative_prompt", "low quality, worst quality, blurry, distorted, deformed")
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)
    num_inference_steps = job_input.get("num_inference_steps", 25)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", -1)
    lora_path = job_input.get("lora_path", "")
    lora_scale = job_input.get("lora_scale", 0.8)
    num_images = job_input.get("num_images", 1)
    
    # Limit num_images to avoid OOM
    num_images = min(num_images, 4)
    
    start_time = time.time()
    
    try:
        images, used_seed = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            lora_path=lora_path,
            lora_scale=lora_scale,
            num_images=num_images
        )
        
        # Convert images to base64
        images_base64 = [image_to_base64(img) for img in images]
        
        generation_time = time.time() - start_time
        
        return {
            "images": images_base64,
            "seed": used_seed,
            "generation_time": generation_time,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "lora_path": lora_path,
                "lora_scale": lora_scale,
                "num_images": num_images
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "generation_time": time.time() - start_time
        }


# RunPod serverless entry point
if __name__ == "__main__":
    import runpod
    
    # Pre-load models during cold start
    print("🚀 Starting RunPod serverless worker...")
    load_models()
    
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
