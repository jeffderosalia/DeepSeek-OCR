"""
Example usage of the DeepSeek-OCR API-style entry point.
Shows how to process PDFs from BytesIO and get markdown results.
"""
from io import BytesIO
from process_pdf_api import process_pdf_to_markdown, DeepSeekOCRProcessor_API


def example_basic_usage():
    """
    Basic example: Load a PDF file and convert to markdown.
    """
    # Load PDF file into BytesIO
    with open('/Users/jeff4work/Downloads/427424.pdf', 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    # Process the PDF
    result = process_pdf_to_markdown(pdf_bytes)

    # Access the results
    print(f"Successfully processed {result.pages_processed} out of {result.total_pages} pages")
    print(f"Pages skipped: {result.pages_skipped}")
    print("\n" + "="*50)
    print("MARKDOWN OUTPUT:")
    print("="*50)
    print(result.markdown)

    # Save markdown to file if needed
    with open('output.md', 'w', encoding='utf-8') as f:
        f.write(result.markdown)


def example_with_image_extraction():
    """
    Example with image extraction: Extract detected images from the PDF.
    """
    with open('your_document.pdf', 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    # Process with image extraction enabled
    result = process_pdf_to_markdown(pdf_bytes, extract_images=True)

    print(f"Extracted {len(result.extracted_images)} images from the document")

    # Save extracted images
    for idx, img in enumerate(result.extracted_images):
        img.save(f'extracted_image_{idx}.jpg')

    # Access the markdown
    print(result.markdown)


def example_custom_configuration():
    """
    Example with custom configuration: Create processor with specific settings.
    """
    # Create processor with custom settings
    processor = DeepSeekOCRProcessor_API(
        max_concurrency=50,  # Lower concurrency for limited GPU memory
        num_workers=32,      # Fewer preprocessing workers
        crop_mode=False,     # Disable dynamic cropping for faster processing
        skip_repeat=True,    # Skip incomplete pages
        prompt="<image>\nFree OCR."  # Use plain OCR without layout detection
    )

    # Load PDF
    with open('your_document.pdf', 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    # Process
    result = processor.process_pdf(pdf_bytes)

    print(result.markdown)


def example_from_http_request():
    """
    Example for web API: Process PDF from HTTP request body.
    """
    # Simulating receiving PDF bytes from a web request
    # In Flask: pdf_bytes = BytesIO(request.data)
    # In FastAPI: pdf_bytes = BytesIO(await file.read())

    # Example with in-memory PDF
    with open('your_document.pdf', 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    # Process without saving to disk
    result = process_pdf_to_markdown(pdf_bytes)

    # Return as JSON-serializable dict
    response = {
        'markdown': result.markdown,
        'pages_processed': result.pages_processed,
        'pages_skipped': result.pages_skipped,
        'total_pages': result.total_pages
    }

    return response


def example_batch_processing():
    """
    Example: Process multiple PDFs in sequence.
    """
    pdf_files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']

    results = []
    for pdf_path in pdf_files:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = BytesIO(f.read())

        result = process_pdf_to_markdown(pdf_bytes)
        results.append({
            'file': pdf_path,
            'markdown': result.markdown,
            'pages': result.pages_processed
        })

    return results


if __name__ == "__main__":
    # Run basic example
    print("Running basic usage example...")
    example_basic_usage()

    # Uncomment to try other examples:
    # example_with_image_extraction()
    # example_custom_configuration()
    # example_from_http_request()
    # example_batch_processing()
