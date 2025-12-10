#!/bin/bash
# Setup script for OCR PII Pipeline

echo "Setting up OCR PII Pipeline..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Please install Python 3.12 first."
    echo "You can install it via Homebrew: brew install python@3.12"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To use the project:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start Jupyter: jupyter notebook ocr_pii_pipeline.ipynb"
echo ""
echo "Note: EasyOCR will download models (~500MB) on first run."

