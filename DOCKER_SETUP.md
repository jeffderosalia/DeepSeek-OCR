# Docker Setup for DeepSeek-OCR

This guide explains how to run DeepSeek-OCR using Docker with GPU support.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (tested with CUDA 11.8)
2. **NVIDIA Docker Runtime** installed
   - Install Docker: https://docs.docker.com/get-docker/
   - Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### 1. Build the Docker Image

```bash
# Navigate to the project directory
cd /path/to/DeepSeek-OCR

# Build using docker-compose (recommended)
docker-compose build

# OR build using docker directly
docker build -t deepseek-ocr:latest .
```

**Note**: Building may take 15-30 minutes due to flash-attention compilation.

### 2. Run the Container

#### Option A: Using docker-compose (Recommended)

```bash
# Start the container in interactive mode
docker-compose run --rm deepseek-ocr
```

#### Option B: Using docker directly

```bash
docker run --rm -it \
  --gpus all \
  --shm-size 16g \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  deepseek-ocr:latest
```

### 3. Run the Example Inside Container

Once inside the container:

```bash
# Test the API with a sample PDF
python test_api.py

# OR run the original PDF script
python run_dpsk_ocr_pdf.py

# OR run the image script
python run_dpsk_ocr_image.py
```

## Using the API

### Prepare Input Files

Create an `input` directory and place your PDF:

```bash
mkdir -p input output
cp your_document.pdf input/
```

### Create a Test Script

Create `test_api.py` in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`:

```python
from io import BytesIO
from process_pdf_api import process_pdf_to_markdown

# Load PDF
with open('/app/input/your_document.pdf', 'rb') as f:
    pdf_bytes = BytesIO(f.read())

# Process
result = process_pdf_to_markdown(pdf_bytes)

# Print results
print(f"Processed {result.pages_processed}/{result.total_pages} pages")
print(f"Pages skipped: {result.pages_skipped}")

# Save markdown
with open('/app/output/result.md', 'w', encoding='utf-8') as f:
    f.write(result.markdown)

print("✓ Markdown saved to /app/output/result.md")
```

### Run in Container

```bash
# Start container
docker-compose run --rm deepseek-ocr

# Inside container, run your script
python test_api.py

# Exit container
exit
```

Check the `output/` directory on your host machine for results.

## Configuration

### Update Config Settings

Before running, edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Set paths
INPUT_PATH = '/app/input/your_document.pdf'
OUTPUT_PATH = '/app/output'

# Choose resolution mode
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# Adjust for your GPU memory
MAX_CONCURRENCY = 100  # Reduce if OOM errors occur
MAX_CROPS = 6          # Reduce to 4 or 2 for limited memory
```

### GPU Memory Issues

If you encounter out-of-memory errors:

1. **Reduce concurrency** in `config.py`:
   ```python
   MAX_CONCURRENCY = 50
   MAX_CROPS = 4
   ```

2. **Lower GPU utilization** in scripts:
   ```python
   gpu_memory_utilization=0.7  # Instead of 0.9
   ```

3. **Use smaller resolution mode**:
   ```python
   BASE_SIZE = 640
   IMAGE_SIZE = 640
   CROP_MODE = False
   ```

## Docker Commands Reference

### Build and Run

```bash
# Build image
docker-compose build

# Run container interactively
docker-compose run --rm deepseek-ocr

# Run with custom command
docker-compose run --rm deepseek-ocr python test_api.py

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Manage Images and Containers

```bash
# List images
docker images

# Remove image
docker rmi deepseek-ocr:latest

# List running containers
docker ps

# List all containers
docker ps -a

# Remove stopped containers
docker container prune
```

### Access Running Container

```bash
# Start container in background
docker-compose up -d

# Execute commands in running container
docker-compose exec deepseek-ocr bash

# Or
docker exec -it deepseek-ocr bash
```

## Volume Mounts

The docker-compose setup includes these volume mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./input` | `/app/input` | Input PDFs/images |
| `./output` | `/app/output` | Output markdown/images |
| `./DeepSeek-OCR-master` | `/app/DeepSeek-OCR-master` | Code (for development) |
| `huggingface-cache` | `/root/.cache/huggingface` | Model cache (persistent) |

## Troubleshooting

### NVIDIA Docker Not Found

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission Denied on Output Files

```bash
# Fix permissions on host
sudo chown -R $USER:$USER output/
```

### Flash Attention Build Errors

If flash-attention fails to build, you can try:
- Using a pre-built wheel
- Skipping flash-attention (model will use regular attention)
- Using a different base image

### Model Download Issues

The model will auto-download from HuggingFace on first run (~10-20GB). Ensure:
- Internet connectivity
- Sufficient disk space
- HuggingFace is accessible

## Production Deployment

For production use:

1. **Remove code mount** from `docker-compose.yml`:
   ```yaml
   # Comment out this line:
   # - ./DeepSeek-OCR-master:/app/DeepSeek-OCR-master
   ```

2. **Use environment variables** instead of editing `config.py`

3. **Set up health checks**

4. **Use a process manager** like supervisord

5. **Implement proper logging**

## Performance Tips

1. **Use SSD storage** for model cache
2. **Increase shared memory** (`shm_size`) for better performance
3. **Pre-download model** to cache volume before production
4. **Use GPU with at least 40GB VRAM** (A100 recommended)
5. **For high throughput**, consider multiple GPU setup with `tensor_parallel_size`

## Next Steps

- See `example_api_usage.py` for more API examples
- Check `CLAUDE.md` for architecture details
- Review `config.py` for all configuration options
