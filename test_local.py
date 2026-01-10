#!/usr/bin/env python3
"""
Local test for the handler without Docker
Run this directly to test the handler functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_handler_locally():
    """Test the handler function directly"""
    print("=" * 50)
    print("🧪 Local Handler Test")
    print("=" * 50)
    
    # Import the handler
    from handler import handler, load_models
    
    print("\n📦 Loading models (this may take a while)...")
    load_models()
    
    print("\n🎨 Testing image generation...")
    
    # Create a test job
    test_job = {
        "id": "local-test-001",
        "input": {
            "prompt": "A majestic mountain landscape at sunset, with snow-capped peaks and a serene lake",
            "negative_prompt": "low quality, blurry",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "seed": 42,
            "lora_path": "",
            "lora_scale": 0.8,
            "num_images": 1
        }
    }
    
    print(f"📝 Prompt: {test_job['input']['prompt'][:50]}...")
    
    # Run the handler
    result = handler(test_job)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n✅ Generation successful!")
    print(f"⏱️ Generation time: {result.get('generation_time', 'N/A'):.2f}s")
    print(f"🎲 Seed used: {result.get('seed', 'N/A')}")
    print(f"🖼️ Images generated: {len(result.get('images', []))}")
    
    # Save the images
    import base64
    os.makedirs("test_output", exist_ok=True)
    
    for i, img_base64 in enumerate(result.get('images', [])):
        filename = f"test_output/local_test_seed{result['seed']}_{i}.png"
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_base64))
        print(f"💾 Saved: {filename}")
    
    print("\n" + "=" * 50)
    print("✅ Local test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_handler_locally()
