"""
Test server for local testing of the FLUX serverless handler
This simulates the RunPod serverless environment locally
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time

# Import the handler
from handler import handler, load_models

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })


@app.route('/runsync', methods=['POST'])
def run_sync():
    """
    Synchronous endpoint - simulates RunPod's /runsync endpoint
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Create job structure like RunPod
        job = {
            "id": f"test-{int(time.time())}",
            "input": data.get("input", data)
        }
        
        print(f"📥 Received job: {job['id']}")
        print(f"📝 Input: {json.dumps(job['input'], indent=2)}")
        
        # Call the handler
        start_time = time.time()
        result = handler(job)
        execution_time = time.time() - start_time
        
        print(f"✅ Job completed in {execution_time:.2f}s")
        
        return jsonify({
            "id": job["id"],
            "status": "COMPLETED",
            "output": result,
            "executionTime": execution_time
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "status": "FAILED",
            "error": str(e)
        }), 500


@app.route('/run', methods=['POST'])
def run_async():
    """
    Async endpoint - for testing, this just runs synchronously
    """
    return run_sync()


@app.route('/', methods=['GET'])
def index():
    """Index page with API info"""
    return jsonify({
        "name": "FLUX Image Generation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/runsync": "POST - Synchronous image generation",
            "/run": "POST - Async image generation (same as runsync for testing)"
        },
        "example_input": {
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
    })


if __name__ == "__main__":
    print("🚀 Starting FLUX Test Server...")
    print("📦 Pre-loading models...")
    
    # Pre-load models
    load_models()
    
    print("✅ Models loaded, starting server...")
    print("🌐 Server running at http://0.0.0.0:8080")
    print("📖 API docs at http://0.0.0.0:8080/")
    
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=False)
