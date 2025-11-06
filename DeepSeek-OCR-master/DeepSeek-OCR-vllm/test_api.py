#!/usr/bin/env python3
"""
Test script for DeepSeek-OCR API.
This script demonstrates how to use the BytesIO-based API.
"""
import os
import sys
from io import BytesIO
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_pdf_api import process_pdf_to_markdown


def find_test_pdf():
    """
    Find a PDF file to test with.
    Checks in order: /app/input, current directory, parent directories
    """
    search_paths = [
        Path('/app/input'),
        Path('.'),
        Path('../..'),
    ]

    for search_path in search_paths:
        if search_path.exists():
            pdf_files = list(search_path.glob('*.pdf'))
            if pdf_files:
                return pdf_files[0]

    return None


def test_basic_api():
    """
    Test basic API functionality with a PDF file.
    """
    print("=" * 70)
    print("DeepSeek-OCR API Test")
    print("=" * 70)

    # Find a PDF to test with
    pdf_path = find_test_pdf()

    if pdf_path is None:
        print("\n❌ No PDF file found for testing.")
        print("\nPlease provide a PDF file in one of these locations:")
        print("  - /app/input/your_document.pdf")
        print("  - ./your_document.pdf")
        print("\nExample:")
        print("  cp /path/to/your/file.pdf /app/input/")
        print("  python test_api.py")
        return

    print(f"\n📄 Found PDF: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load PDF into BytesIO
    print("\n📥 Loading PDF into memory...")
    with open(pdf_path, 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    print("✓ PDF loaded successfully")

    # Process the PDF
    print("\n🔄 Processing PDF with DeepSeek-OCR...")
    print("   (This may take several minutes depending on PDF size)")

    try:
        result = process_pdf_to_markdown(pdf_bytes, extract_images=False)

        # Display results
        print("\n" + "=" * 70)
        print("✓ Processing Complete!")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total pages:       {result.total_pages}")
        print(f"   Pages processed:   {result.pages_processed}")
        print(f"   Pages skipped:     {result.pages_skipped}")
        print(f"   Success rate:      {result.pages_processed/result.total_pages*100:.1f}%")

        # Save output
        output_dir = Path('/app/output')
        if not output_dir.exists():
            output_dir = Path('.')

        # Save markdown
        markdown_path = output_dir / f"{pdf_path.stem}_output.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        print(f"\n💾 Markdown saved to: {markdown_path}")

        # Save raw output with detection tags
        raw_path = output_dir / f"{pdf_path.stem}_raw.md"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(result.markdown_with_detection)
        print(f"💾 Raw output saved to: {raw_path}")

        # Show preview
        print("\n" + "=" * 70)
        print("📝 Markdown Preview (first 500 characters):")
        print("=" * 70)
        preview = result.markdown[:500]
        if len(result.markdown) > 500:
            preview += "..."
        print(preview)

        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


def test_with_image_extraction():
    """
    Test API with image extraction enabled.
    """
    print("=" * 70)
    print("DeepSeek-OCR API Test (With Image Extraction)")
    print("=" * 70)

    pdf_path = find_test_pdf()

    if pdf_path is None:
        print("\n❌ No PDF file found for testing.")
        return

    print(f"\n📄 Processing: {pdf_path}")

    with open(pdf_path, 'rb') as f:
        pdf_bytes = BytesIO(f.read())

    print("\n🔄 Processing with image extraction...")

    try:
        result = process_pdf_to_markdown(pdf_bytes, extract_images=True)

        print("\n✓ Processing Complete!")
        print(f"\n📊 Statistics:")
        print(f"   Pages processed:   {result.pages_processed}")
        print(f"   Images extracted:  {len(result.extracted_images)}")

        # Save extracted images
        output_dir = Path('/app/output')
        if not output_dir.exists():
            output_dir = Path('.')

        images_dir = output_dir / f"{pdf_path.stem}_images"
        images_dir.mkdir(exist_ok=True)

        for idx, img in enumerate(result.extracted_images):
            img_path = images_dir / f"image_{idx:03d}.jpg"
            img.save(img_path)

        if result.extracted_images:
            print(f"\n💾 Images saved to: {images_dir}/")

        # Save markdown
        markdown_path = output_dir / f"{pdf_path.stem}_with_images.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        print(f"💾 Markdown saved to: {markdown_path}")

        print("\n✅ Test with image extraction completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main test function.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Test DeepSeek-OCR API with a PDF file'
    )
    parser.add_argument(
        '--with-images',
        action='store_true',
        help='Enable image extraction'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to specific PDF file to process'
    )

    args = parser.parse_args()

    # Override PDF search if specified
    if args.pdf:
        global find_test_pdf
        original_find = find_test_pdf
        find_test_pdf = lambda: Path(args.pdf) if Path(args.pdf).exists() else None

    if args.with_images:
        test_with_image_extraction()
    else:
        test_basic_api()


if __name__ == "__main__":
    main()
