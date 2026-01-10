#!/usr/bin/env python3
"""
Test client for FLUX serverless endpoint
Usage: python test_client.py [endpoint_url]
"""

import requests
import json
import base64
import sys
import os
from datetime import datetime


def save_image_from_base64(base64_string: str, filename: str):
    """Save base64 encoded image to file"""
    image_data = base64.b64decode(base64_string)
    with open(filename, 'wb') as f:
        f.write(image_data)
    print(f"💾 Saved: {filename}")


def test_endpoint(endpoint_url: str):
    """Test the FLUX serverless endpoint"""
    
    # Test health endpoint
    print("🔍 Testing health endpoint...")
    try:
        health_response = requests.get(f"{endpoint_url}/health", timeout=10)
        print(f"✅ Health check: {health_response.json()}")
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
    
    # Test image generation
    print("\n🎨 Testing image generation...")
    
    test_payload = {
        "input": {
            "prompt": "A majestic mountain landscape at sunset, with snow-capped peaks and a serene lake in the foreground, ultra detailed, 8k",
            "negative_prompt": "low quality, blurry, distorted, deformed, ugly",
            "width": 512,
            "height": 512,
            "num_inference_steps": 25,
            "guidance_scale": 3.5,
            "seed": 42,
            "lora_path": "",
            "lora_scale": 0.8,
            "num_images": 1
        }
    }
    
    print(f"📤 Sending request to {endpoint_url}/runsync")
    print(f"📝 Prompt: {test_payload['input']['prompt'][:50]}...")
    
    try:
        response = requests.post(
            f"{endpoint_url}/runsync",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout for generation
        )
        
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n✅ Generation successful!")
            print(f"⏱️ Execution time: {result.get('executionTime', 'N/A')}s")
            
            output = result.get('output', {})
            
            if 'error' in output:
                print(f"❌ Error in output: {output['error']}")
                return
            
            # Save generated images
            images = output.get('images', [])
            seed = output.get('seed', 'unknown')
            
            print(f"🖼️ Generated {len(images)} image(s) with seed: {seed}")
            
            # Create output directory
            os.makedirs("test_output", exist_ok=True)
            
            for i, img_base64 in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_output/generated_{timestamp}_seed{seed}_{i}.png"
                save_image_from_base64(img_base64, filename)
            
            print("\n📊 Generation parameters:")
            params = output.get('parameters', {})
            for key, value in params.items():
                if key != 'prompt':  # Skip long prompt
                    print(f"   {key}: {value}")
                    
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {result}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {endpoint_url}")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_lora(endpoint_url: str):
    """Test with LoRA weights"""
    print("\n🎨 Testing with LoRA...")
    
    test_payload = {
        "input": {
            "prompt": "A beautiful portrait of a woman",
            "negative_prompt": "low quality, blurry",
            "width": 512,
            "height": 512,
            "num_inference_steps": 25,
            "guidance_scale": 3.5,
            "seed": -1,
            "lora_path": "Joyapeee/juicy-dev",  # Example LoRA
            "lora_scale": 0.8,
            "num_images": 1
        }
    }
    
    print(f"📤 Testing with LoRA: {test_payload['input']['lora_path']}")
    
    try:
        response = requests.post(
            f"{endpoint_url}/runsync",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        
        result = response.json()
        
        if response.status_code == 200 and 'error' not in result.get('output', {}):
            print("✅ LoRA test successful!")
            
            output = result.get('output', {})
            images = output.get('images', [])
            seed = output.get('seed', 'unknown')
            
            os.makedirs("test_output", exist_ok=True)
            
            for i, img_base64 in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_output/lora_test_{timestamp}_seed{seed}_{i}.png"
                save_image_from_base64(img_base64, filename)
        else:
            print(f"⚠️ LoRA test result: {result}")
            
    except Exception as e:
        print(f"❌ LoRA test error: {e}")


if __name__ == "__main__":
    # Default to local test server
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    print("=" * 50)
    print("🧪 FLUX Serverless Endpoint Test")
    print("=" * 50)
    print(f"🌐 Endpoint: {endpoint}")
    print("=" * 50)
    
    # Test basic generation
    test_endpoint(endpoint)
    
    # Optionally test with LoRA
    # test_with_lora(endpoint)
    
    print("\n" + "=" * 50)
    print("✅ Tests completed!")
    print("=" * 50)
