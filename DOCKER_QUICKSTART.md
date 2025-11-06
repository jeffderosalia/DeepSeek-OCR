# Docker Quick Start Guide

## Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- At least 40GB GPU memory (recommended)

## Quick Setup (5 Steps)

### 1. Run Setup Script
```bash
./setup_docker.sh
```

This will:
- Create `input/` and `output/` directories
- Check NVIDIA Docker support
- Build the Docker image (~15-30 minutes)

### 2. Add Your PDF
```bash
cp your_document.pdf input/
```

### 3. Start Container
```bash
docker-compose run --rm deepseek-ocr
```

### 4. Run Test (Inside Container)
```bash
python test_api.py
```

### 5. Check Results
Exit the container and check:
```bash
exit
ls -lh output/
```

## Usage Examples

### Basic API Usage
```python
from io import BytesIO
from process_pdf_api import process_pdf_to_markdown

with open('/app/input/document.pdf', 'rb') as f:
    pdf_bytes = BytesIO(f.read())

result = process_pdf_to_markdown(pdf_bytes)
print(result.markdown)
```

### With Image Extraction
```bash
# Inside container
python test_api.py --with-images
```

### Process Specific File
```bash
# Inside container
python test_api.py --pdf /app/input/specific_file.pdf
```

## Common Commands

```bash
# Build image
docker-compose build

# Run interactively
docker-compose run --rm deepseek-ocr

# Run specific command
docker-compose run --rm deepseek-ocr python test_api.py

# Check logs
docker-compose logs

# Clean up
docker-compose down
docker system prune
```

## Troubleshooting

### Out of Memory Error
Edit `config.py`:
```python
MAX_CONCURRENCY = 50  # Reduce from 100
MAX_CROPS = 4          # Reduce from 6
```

### Slow Processing
- Check GPU usage: `nvidia-smi`
- Reduce `MAX_CROPS` in config.py
- Use smaller resolution mode

### Model Not Downloading
- Check internet connection
- Check disk space (model is ~10-20GB)
- Model caches to: `/root/.cache/huggingface/`

## Next Steps
- See `DOCKER_SETUP.md` for detailed documentation
- See `example_api_usage.py` for more code examples
- See `CLAUDE.md` for architecture details
