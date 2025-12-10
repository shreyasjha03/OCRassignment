#!/usr/bin/env python3
"""Run the complete OCR PII pipeline on all sample images"""

import cv2
import numpy as np
from pathlib import Path
import easyocr
import re
import json
from typing import Dict, List

# Preprocessing functions
def deskew_image(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

def preprocess_image(image_path: str, return_binary: bool = False) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    # Lighter denoising for handwritten text
    denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    if return_binary:
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    return enhanced  # Return grayscale for handwritten text

# OCR
print("Initializing EasyOCR (this may take a moment on first run)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def perform_ocr(image: np.ndarray, use_binary: bool = False):
    # For handwritten text, use more lenient parameters to get more text
    results = reader.readtext(
        image,
        paragraph=False,  # Don't group - get individual detections
        width_ths=0.5,     # Even lower threshold for width (handwritten is variable)
        height_ths=0.5,    # Even lower threshold for height
        detail=1,          # Return detailed results
        allowlist=None,    # Allow all characters
        blocklist=''       # Don't block any characters
    )
    text_lines = []
    for result in results:
        # Handle both tuple and list formats
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            bbox, text, confidence = result[0], result[1], result[2]
            # Very low confidence threshold for handwritten text
            if confidence > 0.1:  # Very low threshold to capture more text
                text_lines.append(text)
    
    # Group into lines based on Y-coordinate
    if text_lines:
        raw_text = ' '.join(text_lines)  # Join with spaces
    else:
        raw_text = ''
    return raw_text, results

# Text cleaning - less aggressive for PII detection
def clean_text(text: str) -> str:
    # Normalize whitespace but keep structure
    text = re.sub(r'\s+', ' ', text)
    # Don't remove too many characters - keep potential PII patterns
    # Only remove clearly problematic OCR artifacts
    text = re.sub(r"[^\w\s\.,;:!?\-'\"()\[\]{}@#%&*+=/\\]", ' ', text)
    text = text.strip()
    return text

# PII Detection - Enhanced for fragmented OCR output
def detect_pii(text: str) -> Dict[str, List[str]]:
    pii = {
        'emails': [],
        'phone_numbers': [],
        'ssn': [],
        'dates': [],
        'names': [],
        'addresses': [],
        'medical_record_numbers': [],
        'dates_of_birth': []
    }
    
    # Normalize text for better matching (remove extra spaces, handle OCR artifacts)
    normalized_text = re.sub(r'\s+', ' ', text)
    
    # Email pattern - more flexible
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
    pii['emails'] = re.findall(email_pattern, normalized_text, re.IGNORECASE)
    
    # Phone numbers - handle fragmented OCR (digits might be separated)
    # Look for sequences of 10+ digits (allowing spaces/dashes)
    phone_text = re.sub(r'[^\d\-\(\)\s]', ' ', normalized_text)
    phone_patterns = [
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 123-456-7890 or 123 456 7890
        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
        r'\d{10}',  # 1234567890
        r'\d{3}\s+\d{3}\s+\d{4}',  # 123 456 7890 (with spaces)
    ]
    for pattern in phone_patterns:
        matches = re.findall(pattern, phone_text)
        # Clean up matches
        cleaned = [re.sub(r'[\s\-\.]', '', m) for m in matches if len(re.sub(r'[\s\-\.\(\)]', '', m)) >= 10]
        pii['phone_numbers'].extend(cleaned)
    pii['phone_numbers'] = list(set([p for p in pii['phone_numbers'] if len(p) >= 10]))
    
    # SSN - handle with/without dashes and spaces
    ssn_text = re.sub(r'[^\d\-\s]', ' ', normalized_text)
    ssn_patterns = [
        r'\d{3}[- ]?\d{2}[- ]?\d{4}',
        r'\d{3}\s+\d{2}\s+\d{4}',
    ]
    for pattern in ssn_patterns:
        matches = re.findall(pattern, ssn_text)
        cleaned = [re.sub(r'[\s\-]', '-', m) for m in matches]
        pii['ssn'].extend(cleaned)
    pii['ssn'] = list(set(pii['ssn']))
    
    # Dates - more flexible patterns
    date_patterns = [
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # 01/15/2024 or 1/15/24
        r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}',  # 2024/01/15
        r'\d{1,2}\s+[/\-]\s+\d{1,2}\s+[/\-]\s+\d{2,4}',  # With spaces
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        pii['dates'].extend(matches)
    pii['dates'] = list(set(pii['dates']))
    
    # Date of Birth - look for keywords with dates nearby
    dob_keywords = r'(?:DOB|Date of Birth|Born|Birth Date|D\.O\.B\.)[: ]*([\d/\-]+|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})'
    dob_matches = re.findall(dob_keywords, normalized_text, re.IGNORECASE)
    pii['dates_of_birth'].extend(dob_matches)
    # Also look for dates near "DOB" or "Birth"
    dob_context = r'(?:DOB|Birth)[\s\S]{0,30}?(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})'
    context_matches = re.findall(dob_context, normalized_text, re.IGNORECASE)
    pii['dates_of_birth'].extend(context_matches)
    pii['dates_of_birth'] = list(set(pii['dates_of_birth']))
    
    # Medical Record Number - more flexible
    mrn_patterns = [
        r'(?:MRN|Medical Record|Record #)[: ]*([A-Z0-9]{4,15})',
        r'MRN\s*[:=]?\s*([A-Z0-9]{4,15})',
        r'Record\s*#?\s*[:=]?\s*([A-Z0-9]{4,15})',
    ]
    for pattern in mrn_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        pii['medical_record_numbers'].extend(matches)
    pii['medical_record_numbers'] = list(set(pii['medical_record_numbers']))
    
    # Names - improved pattern for fragmented OCR
    # Look for capitalized words that might be names
    name_pattern = r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b'
    potential_names = re.findall(name_pattern, normalized_text)
    # Filter out common false positives
    false_positives = {'Date', 'Time', 'Name', 'Address', 'Phone', 'Email', 'Patient', 
                      'Doctor', 'Clinic', 'Hospital', 'Drug', 'Approved', 'Dose', 'Route',
                      'Direction', 'Signature', 'Other'}
    pii['names'] = [name for name in potential_names 
                    if len(name.split()) >= 2  # At least 2 words
                    and not any(fp.lower() in name.lower() for fp in false_positives)
                    and name not in ['Date Time', 'Date Time', 'Other Direction']]
    
    # Addresses - more flexible
    address_patterns = [
        r'\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl)',
        r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:St|Ave|Rd|Dr|Ln|Blvd|Ct|Pl)',
    ]
    for pattern in address_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        pii['addresses'].extend(matches)
    pii['addresses'] = list(set(pii['addresses']))
    
    # Remove empty categories
    pii = {k: v for k, v in pii.items() if v}
    return pii

# Redaction
def create_redacted_image(image_path: str, ocr_results: List, pii_texts: List[str], output_path: str):
    img = cv2.imread(image_path)
    pii_lower = [p.lower() for p in pii_texts]
    for (bbox, text, confidence) in ocr_results:
        text_lower = text.lower()
        if any(pii in text_lower for pii in pii_lower):
            bbox = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(img, [bbox], (0, 0, 0))
    cv2.imwrite(output_path, img)
    return img

# Main pipeline
def process_document(image_path: str, create_redaction: bool = True) -> Dict:
    print(f'\nProcessing: {image_path}')
    print('Step 1: Pre-processing image...')
    # Try OCR on original image first (often works better for handwritten text)
    original_img = cv2.imread(image_path)
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Also preprocess for comparison
    processed_img = preprocess_image(image_path, return_binary=False)
    
    print('Step 2: Performing OCR (this may take a moment)...')
    # Try original image first, fallback to processed if needed
    raw_text, ocr_results = perform_ocr(original_gray, use_binary=False)
    
    # If we didn't get much text, try with processed image
    if len(raw_text) < 50:
        print('  Retrying with preprocessed image...')
        raw_text, ocr_results = perform_ocr(processed_img, use_binary=False)
    
    print('Step 3: Cleaning text...')
    cleaned_text = clean_text(raw_text)
    
    print('Step 4: Detecting PII...')
    pii_detected = detect_pii(cleaned_text)
    
    redacted_path = None
    if create_redaction:
        print('Step 5: Creating redacted image...')
        all_pii_texts = []
        for pii_list in pii_detected.values():
            all_pii_texts.extend(pii_list)
        
        if all_pii_texts:
            base_name = Path(image_path).stem
            output_dir = Path(image_path).parent / 'output'
            output_dir.mkdir(exist_ok=True)
            redacted_path = str(output_dir / f'{base_name}_redacted.jpg')
            create_redacted_image(image_path, ocr_results, all_pii_texts, redacted_path)
        else:
            print('  No PII found to redact')
    
    results = {
        'image_path': image_path,
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'pii_detected': pii_detected,
        'redacted_image_path': redacted_path
    }
    
    return results

# Process all images
if __name__ == '__main__':
    samples_dir = Path('samples')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    image_files = list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.jpeg')) + list(samples_dir.glob('*.JPG')) + list(samples_dir.glob('*.JPEG'))
    
    print(f'Found {len(image_files)} image(s) to process.')
    print('='*60)
    
    all_results = []
    for img_path in image_files:
        results = process_document(str(img_path), create_redaction=True)
        all_results.append(results)
        
        print(f'\nResults for {img_path.name}:')
        print(f'\nRaw Text (first 300 chars):\n{results["raw_text"][:300]}...')
        print(f'\nCleaned Text (first 300 chars):\n{results["cleaned_text"][:300]}...')
        print(f'\nPII Detected:')
        if results['pii_detected']:
            for pii_type, values in results['pii_detected'].items():
                print(f'  {pii_type}: {values}')
        else:
            print('  No PII detected')
        
        if results['redacted_image_path']:
            print(f'\nRedacted image saved to: {results["redacted_image_path"]}')
        
        print('='*60)
    
    # Save results
    results_file = output_dir / 'results.json'
    json_results = []
    for r in all_results:
        json_r = {
            'image_path': str(r['image_path']),
            'raw_text': r['raw_text'],
            'cleaned_text': r['cleaned_text'],
            'pii_detected': r['pii_detected'],
            'redacted_image_path': str(r['redacted_image_path']) if r['redacted_image_path'] else None
        }
        json_results.append(json_r)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f'\n✅ Processing complete!')
    print(f'Results saved to: {results_file}')
    print(f'Redacted images saved in: {output_dir}/')
    print(f'\nTotal images processed: {len(all_results)}')

