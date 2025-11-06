"""
API-style entry point for DeepSeek-OCR PDF processing.
Accepts BytesIO input and returns a DataClass with markdown output.
"""
import os
import fitz
import io
import re
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


@dataclass
class PDFProcessingResult:
    """
    Result of PDF OCR processing.

    Attributes:
        markdown: The extracted markdown content with grounding tags removed
        markdown_with_detection: Raw markdown with detection tags included
        pages_processed: Number of pages successfully processed
        pages_skipped: Number of pages skipped due to incomplete output
        total_pages: Total number of pages in the PDF
    """
    markdown: str
    markdown_with_detection: str
    pages_processed: int
    pages_skipped: int
    total_pages: int
    extracted_images: List[Image.Image] = field(default_factory=list)


class DeepSeekOCRProcessor_API:
    """
    API-style processor for DeepSeek-OCR PDF processing.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_concurrency: int = MAX_CONCURRENCY,
        num_workers: int = NUM_WORKERS,
        crop_mode: bool = CROP_MODE,
        skip_repeat: bool = SKIP_REPEAT,
        prompt: str = PROMPT
    ):
        """
        Initialize the processor with model and configuration.

        Args:
            model_path: Path to the DeepSeek-OCR model
            max_concurrency: Maximum number of concurrent sequences for vLLM
            num_workers: Number of worker threads for image preprocessing
            crop_mode: Whether to use dynamic cropping
            skip_repeat: Whether to skip pages with incomplete outputs
            prompt: The prompt template to use for OCR
        """
        self.max_concurrency = max_concurrency
        self.num_workers = num_workers
        self.crop_mode = crop_mode
        self.skip_repeat = skip_repeat
        self.prompt = prompt

        # Initialize vLLM model
        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            disable_mm_preprocessor_cache=True
        )

        # Configure sampling parameters
        logits_processors = [NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822}
        )]

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        self.image_processor = DeepseekOCRProcessor()

    def _pdf_to_images_from_bytes(
        self,
        pdf_bytes: BytesIO,
        dpi: int = 144,
        image_format: str = "PNG"
    ) -> List[Image.Image]:
        """
        Convert PDF bytes to list of PIL Images.

        Args:
            pdf_bytes: BytesIO object containing PDF data
            dpi: Resolution for image conversion (default 144)
            image_format: Output image format (default PNG)

        Returns:
            List of PIL Image objects, one per page
        """
        images = []

        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes.read(), filetype="pdf")

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None

            if image_format.upper() == "PNG":
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
            else:
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background

            images.append(img)

        pdf_document.close()
        return images

    def _re_match(self, text: str) -> tuple:
        """
        Extract reference tags from OCR output.

        Returns:
            Tuple of (all_matches, image_matches, other_matches)
        """
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)

        matches_image = []
        matches_other = []
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])

        return matches, matches_image, matches_other

    def _extract_images_from_content(
        self,
        content: str,
        image: Image.Image,
        page_idx: int
    ) -> List[Image.Image]:
        """
        Extract cropped images from content based on detection tags.

        Args:
            content: OCR output text with detection tags
            image: Source PIL Image
            page_idx: Page index for naming

        Returns:
            List of cropped PIL Images
        """
        matches_ref, _, _ = self._re_match(content)
        extracted_images = []

        image_width, image_height = image.size
        img_idx = 0

        for ref in matches_ref:
            try:
                label_type = ref[1]
                cor_list = eval(ref[2])

                if label_type == 'image':
                    for points in cor_list:
                        x1, y1, x2, y2 = points

                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            extracted_images.append(cropped)
                        except Exception as e:
                            print(f"Error extracting image: {e}")

                        img_idx += 1
            except Exception as e:
                continue

        return extracted_images

    def _process_single_image(self, image: Image.Image) -> dict:
        """
        Preprocess a single image for vLLM inference.

        Args:
            image: PIL Image to process

        Returns:
            Dictionary with prompt and multimodal data
        """
        cache_item = {
            "prompt": self.prompt,
            "multi_modal_data": {
                "image": self.image_processor.tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=self.crop_mode
                )
            },
        }
        return cache_item

    def process_pdf(self, pdf_file: BytesIO, extract_images: bool = False) -> PDFProcessingResult:
        """
        Process a PDF from BytesIO and return markdown content.

        Args:
            pdf_file: BytesIO object containing the PDF data
            extract_images: Whether to extract detected images from pages (default False)

        Returns:
            PDFProcessingResult dataclass containing markdown and metadata
        """
        # Convert PDF to images
        images = self._pdf_to_images_from_bytes(pdf_file)
        total_pages = len(images)

        # Preprocess images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            batch_inputs = list(tqdm(
                executor.map(self._process_single_image, images),
                total=total_pages,
                desc="Preprocessing images"
            ))

        # Run inference
        outputs_list = self.llm.generate(
            batch_inputs,
            sampling_params=self.sampling_params
        )

        # Post-process outputs
        contents_det = ''
        contents = ''
        extracted_images_list = []
        pages_processed = 0
        pages_skipped = 0

        for page_idx, (output, img) in enumerate(zip(outputs_list, images)):
            content = output.outputs[0].text

            # Check for incomplete output
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
                pages_processed += 1
            else:
                if self.skip_repeat:
                    pages_skipped += 1
                    continue
                pages_processed += 1

            page_separator = f'\n<--- Page Split --->'

            # Store raw content with detection tags
            contents_det += content + f'\n{page_separator}\n'

            # Extract images if requested
            if extract_images:
                page_images = self._extract_images_from_content(content, img, page_idx)
                extracted_images_list.extend(page_images)

            # Process and clean content
            matches_ref, matches_images, matches_other = self._re_match(content)

            # Replace image references (without actual image paths since we're not saving files)
            for idx, a_match_image in enumerate(matches_images):
                # Just remove the detection tags for image references
                content = content.replace(a_match_image, f'[Image {page_idx}_{idx}]')

            # Remove other detection tags and clean up
            for idx, a_match_other in enumerate(matches_other):
                content = content.replace(a_match_other, '').replace(
                    '\\coloneqq', ':='
                ).replace(
                    '\\eqqcolon', '=:'
                ).replace(
                    '\n\n\n\n', '\n\n'
                ).replace(
                    '\n\n\n', '\n\n'
                )

            contents += content + f'\n{page_separator}\n'

        return PDFProcessingResult(
            markdown=contents,
            markdown_with_detection=contents_det,
            pages_processed=pages_processed,
            pages_skipped=pages_skipped,
            total_pages=total_pages,
            extracted_images=extracted_images_list if extract_images else []
        )


# Singleton instance for reusing the loaded model
_processor_instance: Optional[DeepSeekOCRProcessor_API] = None


def get_processor() -> DeepSeekOCRProcessor_API:
    """
    Get or create a singleton processor instance.
    This avoids reloading the model multiple times.

    Returns:
        DeepSeekOCRProcessor_API instance
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DeepSeekOCRProcessor_API()
    return _processor_instance


def process_pdf_to_markdown(pdf_file: BytesIO, extract_images: bool = False) -> PDFProcessingResult:
    """
    Convenience function to process a PDF and extract markdown.

    Args:
        pdf_file: BytesIO object containing the PDF data
        extract_images: Whether to extract detected images from pages (default False)

    Returns:
        PDFProcessingResult dataclass with markdown content and metadata

    Example:
        >>> from io import BytesIO
        >>> with open('document.pdf', 'rb') as f:
        ...     pdf_bytes = BytesIO(f.read())
        >>> result = process_pdf_to_markdown(pdf_bytes)
        >>> print(result.markdown)
        >>> print(f"Processed {result.pages_processed}/{result.total_pages} pages")
    """
    processor = get_processor()
    return processor.process_pdf(pdf_file, extract_images=extract_images)
