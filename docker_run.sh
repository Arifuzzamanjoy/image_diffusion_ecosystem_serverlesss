#!/bin/bash
# ============================================
# Run FLUX Gradio Service in Docker
# ============================================

set -e

IMAGE="flux-gradio:latest"
CONTAINER_NAME="flux-gradio"

echo "=========================================="
echo "🚀 Starting FLUX Gradio Service"
echo "=========================================="

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Create cache directory
mkdir -p cache

# Run the container
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -p 7860:7860 \
    -v $(pwd)/cache:/home/caches \
    -e HF_TOKEN=${HF_TOKEN:-your_hf_token_here} \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    $IMAGE

echo ""
echo "✅ Container started!"
echo ""
echo "📋 Useful commands:"
echo "  View logs:     docker logs -f $CONTAINER_NAME"
echo "  Stop:          docker stop $CONTAINER_NAME"
echo "  Restart:       docker restart $CONTAINER_NAME"
echo ""
echo "🌐 Access the UI at: http://localhost:7860"
echo ""
echo "⏳ Note: First startup takes 5-10 minutes to download models (~30GB)"
echo ""
echo "Following logs..."
docker logs -f $CONTAINER_NAME
