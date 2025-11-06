#!/bin/bash
# Quick setup script for DeepSeek-OCR Docker environment

set -e

echo "=================================="
echo "DeepSeek-OCR Docker Setup"
echo "=================================="
echo ""

# Create necessary directories
echo "📁 Creating input/output directories..."
mkdir -p input output

echo "✓ Directories created:"
echo "  - input/  (place your PDF files here)"
echo "  - output/ (processed results will appear here)"
echo ""

# Check for NVIDIA Docker
echo "🔍 Checking for NVIDIA Docker support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    echo "   Make sure NVIDIA drivers are installed."
    echo ""
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed."
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  Warning: docker-compose not found."
    echo "   You can use 'docker compose' (with space) instead if using Docker Compose V2"
    echo ""
fi

# Test NVIDIA Docker runtime
echo "🔍 Testing NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA Docker runtime is working!"
    echo ""
else
    echo "❌ Error: NVIDIA Docker runtime test failed."
    echo "   Please install NVIDIA Container Toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    exit 1
fi

# Build the Docker image
echo "🏗️  Building Docker image..."
echo "   (This will take 15-30 minutes on first build)"
echo ""

if command -v docker-compose &> /dev/null; then
    docker-compose build
else
    docker compose build
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Place a PDF file in the input/ directory:"
echo "   cp your_document.pdf input/"
echo ""
echo "2. Run the container:"
echo "   docker-compose run --rm deepseek-ocr"
echo ""
echo "3. Inside the container, run the test:"
echo "   python test_api.py"
echo ""
echo "4. Check the output/ directory for results"
echo ""
echo "For more information, see DOCKER_SETUP.md"
echo ""
