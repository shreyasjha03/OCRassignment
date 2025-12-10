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
    # For handwritten text, don't use paragraph mode initially
    # It can cause issues with result format
    results = reader.readtext(
        image,
        paragraph=False,  # Don't group - get individual detections
        width_ths=0.7,    # Lower threshold for width
        height_ths=0.7,   # Lower threshold for height
        detail=1          # Return detailed results
    )
    text_lines = []
    for result in results:
        # Handle both tuple and list formats
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            bbox, text, confidence = result[0], result[1], result[2]
            # Lower confidence threshold for handwritten text
            if confidence > 0.15:  # Even lower for handwritten
                text_lines.append(text)
    
    # Group into lines based on Y-coordinate
    if text_lines:
        raw_text = ' '.join(text_lines)  # Join with spaces for now
    else:
        raw_text = ''
    return raw_text, results

# Text cleaning
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\w\s\.,;:!?\-'\"()\[\]{}@#%&*+=/\\]", '', text)
    text = text.strip()
    return text

# PII Detection
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
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    pii['emails'] = re.findall(email_pattern, text, re.IGNORECASE)
    
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',
        r'\b\d{10}\b'
    ]
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        pii['phone_numbers'].extend(matches)
    pii['phone_numbers'] = list(set(pii['phone_numbers']))
    
    ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
    pii['ssn'] = re.findall(ssn_pattern, text)
    
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        pii['dates'].extend(matches)
    pii['dates'] = list(set(pii['dates']))
    
    dob_keywords = r'\b(?:DOB|Date of Birth|Born|Birth Date)[: ]*([\d/\-]+|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b'
    dob_matches = re.findall(dob_keywords, text, re.IGNORECASE)
    pii['dates_of_birth'] = dob_matches
    
    mrn_patterns = [
        r'\bMRN[: ]*([A-Z0-9]{6,12})\b',
        r'\bMedical Record[: ]*([A-Z0-9]{6,12})\b',
        r'\bRecord #[: ]*([A-Z0-9]{6,12})\b'
    ]
    for pattern in mrn_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        pii['medical_record_numbers'].extend(matches)
    pii['medical_record_numbers'] = list(set(pii['medical_record_numbers']))
    
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'
    potential_names = re.findall(name_pattern, text)
    false_positives = {'Date', 'Time', 'Name', 'Address', 'Phone', 'Email', 'Patient', 'Doctor', 'Clinic', 'Hospital'}
    pii['names'] = [name for name in potential_names if not any(fp in name for fp in false_positives)]
    
    address_pattern = r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl)\b'
    pii['addresses'] = re.findall(address_pattern, text, re.IGNORECASE)
    
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
    processed_img = preprocess_image(image_path, return_binary=False)  # Use grayscale
    
    print('Step 2: Performing OCR (this may take a moment)...')
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

