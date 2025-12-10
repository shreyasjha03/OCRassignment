# OCR Pipeline for Handwritten Document PII Extraction

## Overview
This project implements an end-to-end OCR pipeline for extracting and detecting PII (Personally Identifiable Information) from handwritten documents in JPEG format.

## Pipeline Flow
1. **Pre-processing**: Image rotation correction, noise reduction, contrast enhancement
2. **OCR**: Text extraction using EasyOCR (handwritten text support)
3. **Text Cleaning**: Normalization and artifact removal
4. **PII Detection**: Identifies emails, phone numbers, SSN, dates, names, addresses, medical record numbers
5. **Redaction** (Optional): Creates redacted images with PII blacked out

## Setup

### Prerequisites
- Python 3.12 (or 3.11) - Python 3.13 is not yet supported by PyTorch
- pip

### Installation

1. Create a virtual environment (recommended):
```bash
python3.12 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Note**: EasyOCR will automatically download required models on first run (~500MB, may take a few minutes).

## Usage

1. **Add sample images**: Place your JPEG images in the `samples/` folder

2. **Activate virtual environment and run the notebook**:
```bash
source venv/bin/activate  # If not already activated
jupyter notebook ocr_pii_pipeline.ipynb
```

3. **Execute all cells**: The notebook will:
   - Process all images in the `samples/` folder
   - Extract text
   - Detect PII
   - Generate redacted images (saved in `output/` folder)
   - Save results to `output/results.json`

## Output

- **Redacted images**: Saved in `output/` folder with `_redacted.jpg` suffix
- **Results JSON**: `output/results.json` contains all extracted text and detected PII

## Features

- Handles slightly tilted images (automatic deskewing)
- Works with different handwriting styles
- Optimized for doctor/clinic-style notes and forms
- Detects multiple PII types:
  - Email addresses
  - Phone numbers
  - Social Security Numbers (SSN)
  - Dates and dates of birth
  - Names
  - Addresses
  - Medical Record Numbers (MRN)

## Dependencies

See `requirements.txt` for complete list. Main libraries:
- `opencv-python`: Image preprocessing
- `easyocr`: OCR engine (handwritten text support)
- `numpy`: Image processing
- `Pillow`: Image manipulation
- `matplotlib`: Visualization
- `jupyter`: Notebook environment

## Notes

- First run will download EasyOCR models (~500MB)
- Processing time depends on image size and complexity
- PII detection uses regex patterns - may have false positives/negatives
- For production use, consider using more advanced NER models for name detection

