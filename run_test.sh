#!/bin/bash
# Build and Run Test Server for local testing

set -e

echo "=========================================="
echo "🧪 FLUX Test Server Build & Run Script"
echo "=========================================="

IMAGE_NAME="flux-test"
CONTAINER_NAME="flux-test"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create directories
mkdir -p cache output test_output

echo -e "${GREEN}📦 Building test Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest -f Dockerfile.test .

echo ""
echo -e "${GREEN}🚀 Starting test server...${NC}"

# Stop existing container if running
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Run container
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p 8080:8080 \
    -v $(pwd)/cache:/runpod-volume/cache \
    -v $(pwd)/output:/app/output \
    -e HF_TOKEN=${HF_TOKEN:-your_hf_token_here} \
    ${IMAGE_NAME}:latest

echo ""
echo -e "${GREEN}✅ Test server starting...${NC}"
echo ""
echo "📋 Useful commands:"
echo "  View logs:     docker logs -f ${CONTAINER_NAME}"
echo "  Stop server:   docker stop ${CONTAINER_NAME}"
echo "  Test endpoint: python test_client.py http://localhost:8080"
echo ""
echo -e "${YELLOW}⏳ Waiting for server to be ready (models loading...)${NC}"
echo "   This may take several minutes for the first run."
echo ""
echo "Follow logs with: docker logs -f ${CONTAINER_NAME}"
