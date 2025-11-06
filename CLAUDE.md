# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSeek-OCR is a multimodal AI model for optical character recognition and document understanding. The model combines vision encoders with LLMs to extract text and layouts from images and PDFs with high compression efficiency. The project supports both vLLM-based and Transformers-based inference implementations.

## Environment Setup

**Required Environment:**
- CUDA 11.8 + PyTorch 2.6.0
- Python 3.12.9

**Installation Commands:**
```bash
# Create conda environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download vLLM wheel from https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install dependencies
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** The vLLM/transformers version conflict warning can be ignored when running both implementations in the same environment.

## Running Inference

### vLLM Implementation (Faster, Batch Processing)

All vLLM scripts are located in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`.

**Before running, configure `config.py`:**
- Set `INPUT_PATH` and `OUTPUT_PATH`
- Choose resolution mode by setting `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`
- Adjust `MAX_CONCURRENCY` and `NUM_WORKERS` based on GPU memory
- Select appropriate `PROMPT` for task type

**Run commands:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Single image with streaming output
python run_dpsk_ocr_image.py

# PDF processing with concurrency (~2500 tokens/s on A100-40G)
python run_dpsk_ocr_pdf.py

# Batch evaluation for benchmarks
python run_dpsk_ocr_eval_batch.py
```

### Transformers Implementation (Simpler, HuggingFace-based)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

Or use in Python:
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR',
                                   _attn_implementation='flash_attention_2',
                                   trust_remote_code=True)
model.infer(tokenizer, prompt=prompt, image_file=image_file,
            base_size=1024, image_size=640, crop_mode=True)
```

## Architecture

### Two Inference Paths

1. **vLLM Path** (`DeepSeek-OCR-vllm/`): Custom vLLM integration with optimized batch processing
   - Custom model registration via `ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)`
   - N-gram no-repeat logits processor to prevent text repetition
   - Concurrent PDF processing with ThreadPoolExecutor
   - Dynamic image preprocessing with crop ratio calculation

2. **Transformers Path** (`DeepSeek-OCR-hf/`): Standard HuggingFace interface using `model.infer()` method

### Key Components

**Vision Encoders** (`deepencoder/`):
- `sam_vary_sdpa.py`: SAM-based vision encoder
- `clip_sdpa.py`: CLIP-L vision encoder
- `build_linear.py`: MLP projector connecting vision to language

**Image Processing** (`process/image_process.py`):
- `DeepseekOCRProcessor`: Handles image preprocessing and tokenization
- `dynamic_preprocess()`: Splits large images into tiles based on aspect ratio
- `count_tiles()`: Calculates optimal crop ratios (configurable MIN_CROPS=2, MAX_CROPS=6)

**Model Core** (`deepseek_ocr.py`):
- Custom vLLM model implementation
- Multimodal input handling and embeddings
- Integration with vLLM's sampling and profiling systems

**Text Processing** (`process/ngram_norepeat.py`):
- `NoRepeatNGramLogitsProcessor`: Prevents repetitive output with configurable ngram size and window
- Whitelist tokens for HTML table tags (`<td>`, `</td>`)

### Resolution Modes

The model supports multiple resolution modes configured via `BASE_SIZE`, `IMAGE_SIZE`, and `CROP_MODE` in `config.py`:

- **Tiny**: 512×512 (64 vision tokens) - `base_size=512, image_size=512, crop_mode=False`
- **Small**: 640×640 (100 vision tokens) - `base_size=640, image_size=640, crop_mode=False`
- **Base**: 1024×1024 (256 vision tokens) - `base_size=1024, image_size=1024, crop_mode=False`
- **Large**: 1280×1280 (400 vision tokens) - `base_size=1280, image_size=1280, crop_mode=False`
- **Gundam** (dynamic): n×640×640 + 1×1024×1024 - `base_size=1024, image_size=640, crop_mode=True`

## Prompt Engineering

Choose prompts based on task type (examples in `config.py`):

```python
# Document to markdown conversion
"<image>\n<|grounding|>Convert the document to markdown."

# General OCR with layout detection
"<image>\n<|grounding|>OCR this image."

# Plain text extraction without layouts
"<image>\nFree OCR."

# Figure/chart parsing
"<image>\nParse the figure."

# General image description
"<image>\nDescribe this image in detail."

# Object location/grounding
"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."
```

## Output Processing

The model produces grounding outputs with reference tags: `<|ref|>label<|/ref|><|det|>coordinates<|/det|>`

**Post-processing** (in `run_dpsk_ocr_image.py` and `run_dpsk_ocr_pdf.py`):
- `re_match()`: Extracts reference tags for images and other elements
- `draw_bounding_boxes()`: Visualizes detected regions
- Image elements are cropped and saved to `OUTPUT_PATH/images/`
- Markdown output with embedded image references is saved to `.mmd` files

## Performance Tuning

**GPU Memory Management** (`config.py`):
- `MAX_CROPS`: Maximum tiles per image (default 6, max 9) - reduce for limited memory
- `MAX_CONCURRENCY`: Batch size for PDF processing (default 100)
- `NUM_WORKERS`: Image preprocessing threads (default 64)
- vLLM `gpu_memory_utilization`: 0.75 for streaming, 0.9 for batch

**Special Configuration**:
- Set `TRITON_PTXAS_PATH` for CUDA 11.8 compatibility
- Set `VLLM_USE_V1='0'` to use legacy vLLM engine
- Use `SKIP_REPEAT=True` to skip pages with incomplete outputs

## Model Download

The model is hosted on HuggingFace: `deepseek-ai/DeepSeek-OCR`

It will auto-download when first running inference, or set `MODEL_PATH` in `config.py` to a local path.
