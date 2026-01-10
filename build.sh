#!/bin/bash
# Build and Run FLUX Serverless Docker Container

set -e

echo "=========================================="
echo "🐳 FLUX Serverless Docker Build Script"
echo "=========================================="

# Configuration
IMAGE_NAME="flux-serverless"
IMAGE_TAG="latest"
CONTAINER_NAME="flux-serverless"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}⚠️ NVIDIA Docker runtime may not be configured${NC}"
fi

# Create cache directory
mkdir -p cache output

echo ""
echo -e "${GREEN}📦 Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

echo ""
echo -e "${GREEN}✅ Build completed!${NC}"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 8000:8000 -v \$(pwd)/cache:/runpod-volume/cache ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} your-dockerhub-username/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  docker push your-dockerhub-username/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "For RunPod deployment, use the image: your-dockerhub-username/${IMAGE_NAME}:${IMAGE_TAG}"
