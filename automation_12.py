import os
import torch
import random
import json
import time
import tempfile
import gradio as gr
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
from huggingface_hub import login
from dotenv import load_dotenv


# Set up optimal cache location for runpod
CACHE_DIR = os.path.expanduser("/home/caches")
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

# Login to Huggingface
# Login to Huggingface

load_dotenv()  # Loads variables from .env

# Login to Huggingface
#hf_token = os.getenv('HF_TOKEN')
#if hf_token:
#    login(token=hf_token)
#else:
#    print("⚠️ HF_TOKEN not found in .env file. Please add it for model downloads.")

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

# Image generation function with LoRA support
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
    num_images=1
):
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
            saved_paths = save_images_to_files(images)
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

# Batch image generation from JSON file
def generate_images_from_json(
    json_file_path,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    lora_path,
    lora_scale,
    output_dir
):
    try:
        # Load prompts from JSON file
        with open(json_file_path, 'r') as f:
            prompts = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
        seeds_used = []
        
        # Process each prompt in the JSON file
        for i, item in enumerate(prompts):
            prompt = item['prompt'] if isinstance(item, dict) and 'prompt' in item else item
            
            # Generate a random seed for each image
            seed = random.randint(0, 2147483647)
            seeds_used.append(seed)
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Configure optimal parameters
            safe_width = (width // 16) * 16
            safe_height = (height // 16) * 16
            safe_guidance = min(guidance_scale, 12.0)
            
            # Enhanced prompt
            full_prompt = f"  {prompt} "
            
            print(f"🔄 Generating image {i+1}/{len(prompts)} with seed: {seed}")
            
            try:
                # Direct approach without encoding
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
              #      joint_attention_kwargs=joint_attention_kwargs,
                    output_type="pil"
                ).images
                
                if images:
                    # Save each image with proper naming
                    for j, image in enumerate(images):
                        image_path = os.path.join(output_dir, f"prompt_{i}_image_{j}_seed_{seed}.png")
                        image.save(image_path)
                        all_image_paths.append(image_path)
                    
                    generated_images.extend(images)
                    
                    # Provide update after each successful generation
                    yield generated_images, f"Generated {i+1}/{len(prompts)} images", all_image_paths, seeds_used
            
            except Exception as e:
                print(f"❌ Generation failed for prompt {i+1}: {e}")
                continue
        
        return generated_images, f"Completed generating {len(generated_images)} images", all_image_paths, seeds_used
    
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return [], f"Error: {str(e)}", [], []

# Gradio UI
with gr.Blocks(title="FLUX Image Generator") as demo:
    gr.Markdown("# FLUX Image Generator")
    gr.Markdown("Generate high-quality images using the FLUX model with optional LoRA weights")
    
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want to see in the image",
                    lines=2,
                    value="low quality, worst quality, blurry, distorted, deformed"
                )
                
                # LoRA settings
                lora_path = gr.Textbox(
                    label="LoRA Path",
                    placeholder="e.g., username/model-name (leave empty for no LoRA)",
                    value="Joyapeee/juicy-dev"
                )
                
                lora_scale = gr.Slider(
                    label="LoRA Scale", 
                    minimum=0, 
                    maximum=3.0, 
                    value=0.8, 
                    step=0.05,
                    info="Controls how strongly the LoRA adapters are applied"
                )
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=384, maximum=1024, value=512, step=64)
                    height = gr.Slider(label="Height", minimum=384, maximum=1024, value=512, step=64)
                
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=15, maximum=500, value=25, step=1)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=12, value=3.5, step=0.1)
                
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    num_images = gr.Slider(label="Number of Images", minimum=1, maximum=4, value=1, step=1)
                
                generate_btn = gr.Button("Generate Images", variant="primary")
                
            with gr.Column(scale=1):
                # Output gallery
                output_images = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery")
                used_seed = gr.Number(label="Used Seed", visible=True)
                
                # Add download files component
                download_files = gr.Files(label="Download Images")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear")
    
    with gr.Tab("Batch Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls for batch processing
                json_file = gr.Textbox(
                    label="JSON File Path",
                    placeholder="Path to your JSON file with prompts",
                    value="/workspace/prompt.json"
                )
                
                batch_negative_prompt = gr.Textbox(
                    label="Negative Prompt for Batch",
                    placeholder="What you don't want to see in the images",
                    lines=2,
                    value="low quality, worst quality, blurry, distorted, deformed"
                )
                
                # LoRA settings for batch
                batch_lora_path = gr.Textbox(
                    label="LoRA Path for Batch",
                    placeholder="e.g., username/model-name (leave empty for no LoRA)",
                    value="Joyapeee/juicy-dev"
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
                    value="/workspace/batch_output"
                )
                
                with gr.Row():
                    batch_width = gr.Slider(label="Width", minimum=384, maximum=1024, value=512, step=64)
                    batch_height = gr.Slider(label="Height", minimum=384, maximum=1024, value=512, step=64)
                
                with gr.Row():
                    batch_steps = gr.Slider(label="Steps", minimum=15, maximum=300, value=25, step=1)
                    batch_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=12, value=3.5, step=0.1)
                
                batch_generate_btn = gr.Button("Generate Batch", variant="primary")
                
            with gr.Column(scale=1):
                # Output gallery for batch
                batch_output_images = gr.Gallery(label="Batch Generated Images", show_label=True)
                batch_status = gr.Textbox(label="Batch Status", value="Ready")
                batch_download_files = gr.Files(label="Download Batch Images")
                batch_seeds = gr.Textbox(label="Seeds Used", value="")
                
                with gr.Row():
                    batch_clear_btn = gr.Button("Clear Batch Results")
    
    with gr.Tab("Help"):
        gr.Markdown("""
        ## Tips for Better Results
        
        * Be specific in your prompts. Include details about style, lighting, mood, etc.
        * Use the negative prompt to exclude unwanted elements.
        * Higher guidance scale means stronger adherence to the prompt, but possibly less creativity.
        * More steps generally means better quality but longer generation time.
        * Save the seed of images you like so you can reproduce them later with variations.
        
        ## Using LoRAs
        
        * Enter the Hugging Face model ID in the LoRA Path field (e.g., "username/model-name")
        * Adjust the LoRA scale to control the strength (usually 0.6-0.8 works well)
        * Leave the LoRA path empty to use the base model without LoRA
        
        ## Batch Generation
        
        * Prepare a JSON file with prompts. The file should be a list of objects with a "prompt" field or simple strings.
        * Example JSON format:
          ```json
          [
            {"prompt": "A beautiful landscape with mountains"},
            {"prompt": "A cat sitting on a windowsill"},
            "A futuristic cityscape at night"
          ]
          ```
        * Specify the output directory to save all the generated images
        * Each image filename will include the prompt index and the seed used
        
        ## About This Model
        
        This model uses FLUX.1-dev, which specializes in photorealistic results with optional LoRA weights.
        
        ## Troubleshooting
        
        * If you encounter errors, try decreasing the image size or guidance scale
        * Increase steps for better quality but longer generation time
        * Try different seeds if you don't like the result
        """
        )
    
    # Set up event handlers
    def clear_outputs():
        return [None, None, None]
    
    def clear_batch_outputs():
        return [None, "Ready", None, ""]
    
    # Single image generation
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt, 
            negative_prompt, 
            width, 
            height, 
            steps, 
            guidance_scale, 
            seed,
            lora_path,
            lora_scale,
            num_images
        ],
        outputs=[output_images, used_seed, download_files]
    )
    
    # Batch generation
    batch_generate_btn.click(
        fn=generate_images_from_json,
        inputs=[
            json_file,
            batch_negative_prompt,
            batch_width,
            batch_height,
            batch_steps,
            batch_guidance_scale,
            batch_lora_path,
            batch_lora_scale,
            output_directory
        ],
        outputs=[batch_output_images, batch_status, batch_download_files, batch_seeds]
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
        outputs=[batch_output_images, batch_status, batch_download_files, batch_seeds]
    )

# Launch the app
# Use environment variables for server configuration (Docker-friendly)
import os
server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

demo.launch(
    server_name=server_name,
    server_port=server_port,
    share=share,
    quiet=True
)