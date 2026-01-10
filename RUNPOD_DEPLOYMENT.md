# FLUX Serverless Deployment Guide for RunPod

## Overview

This guide explains how to deploy the FLUX image generation service as a serverless endpoint on RunPod.

## Files Created

- `handler.py` - RunPod serverless handler (main entry point)
- `Dockerfile` - Docker image for RunPod serverless deployment
- `Dockerfile.test` - Docker image for local testing with HTTP endpoint
- `docker-compose.yml` - Docker Compose configuration
- `requirements_serverless.txt` - Python dependencies
- `test_server.py` - Local HTTP test server
- `test_client.py` - Client script to test the endpoint
- `build.sh` - Build script for Docker image
- `run_test.sh` - Script to run local test server

## Building the Docker Image

### Option 1: Build Locally

```bash
# Build the RunPod serverless image
docker build -t flux-serverless:latest -f Dockerfile .

# Build the test image (with HTTP server)
docker build -t flux-test:latest -f Dockerfile.test .
```

### Option 2: Build on a Cloud VM

If you don't have a local GPU, you can build on a cloud VM:

```bash
# SSH to your cloud VM with GPU
ssh your-vm

# Clone or copy the files
git clone your-repo
cd your-repo

# Build the image
docker build -t flux-serverless:latest -f Dockerfile .
```

## Pushing to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag flux-serverless:latest YOUR_DOCKERHUB_USERNAME/flux-serverless:latest

# Push the image
docker push YOUR_DOCKERHUB_USERNAME/flux-serverless:latest
```

## Deploying to RunPod

### Step 1: Create a RunPod Account
Go to https://runpod.io and create an account.

### Step 2: Create a Serverless Endpoint

1. Go to **Serverless** in the RunPod dashboard
2. Click **New Endpoint**
3. Configure:
   - **Name**: FLUX Image Generation
   - **Docker Image**: `YOUR_DOCKERHUB_USERNAME/flux-serverless:latest`
   - **GPU Type**: Select a GPU with at least 24GB VRAM (A10G, A100, etc.)
   - **Environment Variables**:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
4. Click **Create**

### Step 3: Configure Network Volume (Recommended)

For faster cold starts, attach a network volume to cache models:

1. Create a network volume (at least 50GB)
2. Mount it at `/runpod-volume`
3. Models will be cached and persist between runs

## Using the Endpoint

### API Endpoint

Your endpoint URL will be:
```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
```

### Authentication

Add your RunPod API key to the request headers:
```
Authorization: Bearer {YOUR_RUNPOD_API_KEY}
```

### Request Format

```json
{
  "input": {
    "prompt": "A beautiful sunset over mountains",
    "negative_prompt": "low quality, blurry",
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 3.5,
    "seed": -1,
    "lora_path": "",
    "lora_scale": 0.8,
    "num_images": 1
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | The text prompt for image generation |
| `negative_prompt` | string | "low quality..." | What to avoid in the image |
| `width` | int | 512 | Image width (must be divisible by 16) |
| `height` | int | 512 | Image height (must be divisible by 16) |
| `num_inference_steps` | int | 25 | Number of denoising steps |
| `guidance_scale` | float | 3.5 | How closely to follow the prompt |
| `seed` | int | -1 | Random seed (-1 for random) |
| `lora_path` | string | "" | HuggingFace path to LoRA weights |
| `lora_scale` | float | 0.8 | Strength of LoRA weights |
| `num_images` | int | 1 | Number of images to generate (max 4) |

### Response Format

```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "output": {
    "images": ["base64_encoded_image_1", "base64_encoded_image_2"],
    "seed": 42,
    "generation_time": 15.5,
    "parameters": {
      "prompt": "...",
      "width": 512,
      "height": 512
    }
  }
}
```

### Example: Python Client

```python
import requests
import base64

API_KEY = "your_runpod_api_key"
ENDPOINT_ID = "your_endpoint_id"

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "prompt": "A beautiful sunset over mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 25
        }
    }
)

result = response.json()
images = result["output"]["images"]

# Save the first image
with open("output.png", "wb") as f:
    f.write(base64.b64decode(images[0]))
```

### Example: cURL

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains",
      "width": 512,
      "height": 512,
      "num_inference_steps": 25
    }
  }'
```

## Local Testing

### Using Docker Compose

```bash
# Start the test server
docker-compose --profile test up flux-test

# In another terminal, test the endpoint
python test_client.py http://localhost:8080
```

### Using the Test Script

```bash
# Build and run test server
./run_test.sh

# Test the endpoint
python test_client.py http://localhost:8080
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce image dimensions (512x512 is recommended)
- Reduce `num_images` to 1
- Use a GPU with more VRAM

### Slow Cold Starts
- Use a network volume to cache models
- Models are ~30GB and take 5-10 minutes to download on first run

### LoRA Loading Issues
- Ensure the LoRA path is correct (format: `username/model-name`)
- Check if the LoRA is compatible with FLUX.1-dev

### HuggingFace Token Issues
- Ensure `HF_TOKEN` environment variable is set
- The token needs read access to gated models

## Model Information

This service uses:
- **Base Model**: FLUX.1-dev by Black Forest Labs
- **Text Encoder 1**: CLIP ViT-L/14@336px
- **Text Encoder 2**: T5-v1.1-XXL
- **VAE**: FLUX.1-dev VAE

## License

This deployment is subject to the FLUX.1-dev license terms from Black Forest Labs.
