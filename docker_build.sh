#!/bin/bash
# ============================================
# FLUX Image Generation - Docker Build Script
# ============================================
# Run this script on a machine with Docker installed
# 
# Usage: ./docker_build.sh [TAG]
# Example: ./docker_build.sh myusername/flux-gradio:v1

set -e

# Configuration
DEFAULT_TAG="flux-gradio:latest"
TAG="${1:-$DEFAULT_TAG}"

echo "=========================================="
echo "🐳 Building FLUX Gradio Docker Image"
echo "=========================================="
echo "Tag: $TAG"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo ""
    echo "Please install Docker first:"
    echo "  Ubuntu/Debian: sudo apt-get install docker.io"
    echo "  Or visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✅ NVIDIA Docker runtime detected"
else
    echo "⚠️ NVIDIA Docker runtime not detected"
    echo "   GPU support may not work. Install nvidia-docker2 for GPU support."
fi

# Create required directories
mkdir -p cache output

# Build the image
echo ""
echo "📦 Building Docker image..."
docker build -t "$TAG" -f Dockerfile .

echo ""
echo "=========================================="
echo "✅ Build Complete!"
echo "=========================================="
echo ""
echo "To run the container locally:"
echo "  docker run --gpus all -p 7860:7860 -v \$(pwd)/cache:/home/caches $TAG"
echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push $TAG"
echo ""
echo "For RunPod deployment:"
echo "  1. Push image to Docker Hub"
echo "  2. Create a new Pod/Template on RunPod"
echo "  3. Use image: $TAG"
echo "  4. Expose port 7860"
echo "  5. Set environment variable HF_TOKEN"
