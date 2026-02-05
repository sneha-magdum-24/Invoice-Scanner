import cv2
import numpy as np
import re
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
import requests
import json
import random
from typing import Dict, List, Optional, Tuple
import logging

# Optional imports with fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available")

try:
    import fitz  # PyMuPDF for PDF processing
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("Warning: PyMuPDF (fitz) not available - PDF processing disabled")

class EnhancedInvoiceOCR:
    """
    Advanced OCR+LLM system optimized for blurry and poorly captured invoice images.
    Features:
    - Multi-stage image preprocessing for blur and noise reduction
    - Dual OCR engine support (EasyOCR + Tesseract)
    - Advanced text reconstruction and cleaning
    - LLM-powered structured data extraction
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.setup_logging()
        
        # Initialize OCR readers
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.log("EasyOCR initialized successfully")
            except Exception as e:
                self.log(f"EasyOCR initialization failed: {e}")
        
        # OCR confidence thresholds
        self.easyocr_threshold = 0.3
        self.tesseract_threshold = 30
    
    def setup_logging(self):
        """Setup logging for debug output"""
        logging.basicConfig(level=logging.INFO if self.debug else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def log(self, message):
        """Log debug messages"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def advanced_image_preprocessing(self, image_path: str) -> List[np.ndarray]:
        """
        Advanced multi-stage image preprocessing optimized for blurry/poor quality images.
        Returns multiple processed versions for ensemble OCR.
        """
        self.log(f"Starting advanced preprocessing for: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        self.log(f"Original image size: {original_width}x{original_height}")
        
        processed_images = []
        
        # 1. Resize for optimal OCR (if needed)
        if original_width > 3000 or original_height > 3000:
            scale = min(3000/original_width, 3000/original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            self.log(f"Resized to: {new_width}x{new_height}")
        elif original_width < 1000:
            # Upscale small images
            scale = 1500 / original_width
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            self.log(f"Upscaled to: {new_width}x{new_height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Blur reduction using multiple techniques
        
        # Method 1: Unsharp masking for blur reduction
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        processed_images.append(("unsharp_mask", unsharp_mask))
        
        # Method 2: Wiener-like deconvolution approximation
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(("sharpened", sharpened))
        
        # Method 3: CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        processed_images.append(("clahe", clahe_enhanced))
        
        # 3. Noise reduction
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed_images.append(("denoised", denoised))
        
        # 4. Multiple thresholding approaches
        
        final_processed = []
        
        for name, processed_img in processed_images:
            # Adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(
                processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Otsu's threshold
            _, otsu_thresh = cv2.threshold(
                processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            adaptive_cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            otsu_cleaned = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
            
            final_processed.extend([
                (f"{name}_adaptive", adaptive_cleaned),
                (f"{name}_otsu", otsu_cleaned)
            ])
        
        self.log(f"Generated {len(final_processed)} processed image variants")
        return final_processed
    
    def extract_text_ensemble(self, image_path: str) -> str:
        """
        Extract text using ensemble of preprocessing methods and OCR engines.
        """
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(image_path)
        
        # Get multiple processed versions
        processed_images = self.advanced_image_preprocessing(image_path)
        
        all_extractions = []
        
        # Try each processed image with available OCR engines
        for name, processed_img in processed_images:
            self.log(f"Processing variant: {name}")
            
            # EasyOCR extraction with layout preservation
            if self.easyocr_reader:
                try:
                    easyocr_text = self.extract_with_easyocr_detailed(processed_img)
                    if easyocr_text:
                        all_extractions.append((f"{name}_easyocr", easyocr_text))
                except Exception as e:
                    self.log(f"EasyOCR failed for {name}: {e}")
            
            # Tesseract extraction with layout
            if TESSERACT_AVAILABLE:
                try:
                    tesseract_text = self.extract_with_tesseract_detailed(processed_img)
                    if tesseract_text:
                        all_extractions.append((f"{name}_tesseract", tesseract_text))
                except Exception as e:
                    self.log(f"Tesseract failed for {name}: {e}")
        
        if not all_extractions:
            self.log("No successful text extractions")
            return ""
        
        # Select best extraction based on content quality
        best_text = self.select_best_extraction(all_extractions)
        self.log(f"Selected best extraction with {len(best_text)} characters")
        
        return best_text
    
    def extract_with_easyocr_detailed(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR with detailed layout preservation"""
        if not self.easyocr_reader:
            return ""
        
        results = self.easyocr_reader.readtext(image)
        high_conf_results = [r for r in results if r[2] > self.easyocr_threshold]
        
        if not high_conf_results:
            return ""
        
        return self.reconstruct_table_structure(high_conf_results)
    
    def extract_with_tesseract_detailed(self, image: np.ndarray) -> str:
        """Extract text using Tesseract with detailed layout preservation"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Filter high confidence text
            high_conf_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > self.tesseract_threshold and data['text'][i].strip():
                    high_conf_data.append({
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': data['conf'][i]
                    })
            
            return self.reconstruct_table_from_tesseract(high_conf_data)
            
        except Exception as e:
            self.log(f"Tesseract detailed extraction failed: {e}")
            return ""
    
    def reconstruct_table_structure(self, ocr_results: List) -> str:
        """Reconstruct table structure preserving columns for quantity, price, amount"""
        if not ocr_results:
            return ""
        
        # Sort by Y coordinate first (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1] if isinstance(x[0], list) else 0)
        
        # Group into rows based on Y coordinate
        rows = []
        current_row = []
        y_threshold = 15  # Pixels
        
        for item in sorted_results:
            bbox, text, conf = item
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
                
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            
            if not current_row:
                current_row.append((bbox, text, conf))
            else:
                # Check if this item belongs to current row
                row_y = sum((b[0][1] + b[2][1]) / 2 for b, t, c in current_row) / len(current_row)
                
                if abs(y_center - row_y) <= y_threshold:
                    current_row.append((bbox, text, conf))
                else:
                    # Sort current row by X coordinate and add to rows
                    current_row.sort(key=lambda x: x[0][0][0])
                    rows.append(current_row)
                    current_row = [(bbox, text, conf)]
        
        if current_row:
            current_row.sort(key=lambda x: x[0][0][0])
            rows.append(current_row)
        
        # Reconstruct text with proper spacing
        final_text = ""
        for row in rows:
            row_text = ""
            prev_x_end = 0
            
            for bbox, text, conf in row:
                x_start = bbox[0][0]
                
                # Add spacing based on gap
                if prev_x_end > 0:
                    gap = x_start - prev_x_end
                    if gap > 50:  # Large gap - likely new column
                        row_text += "\t\t"
                    elif gap > 20:  # Medium gap
                        row_text += "\t"
                    else:
                        row_text += " "
                
                row_text += text
                prev_x_end = bbox[1][0]  # Right edge
            
            final_text += row_text.strip() + "\n"
        
        return final_text
    
    def reconstruct_table_from_tesseract(self, data: List[Dict]) -> str:
        """Reconstruct table from Tesseract data with proper column alignment"""
        if not data:
            return ""
        
        # Sort by top coordinate first
        data.sort(key=lambda x: x['top'])
        
        # Group into rows
        rows = []
        current_row = []
        y_threshold = 10
        
        for item in data:
            if not current_row:
                current_row.append(item)
            else:
                # Check if belongs to current row
                row_y = sum(x['top'] for x in current_row) / len(current_row)
                if abs(item['top'] - row_y) <= y_threshold:
                    current_row.append(item)
                else:
                    # Sort current row by left coordinate
                    current_row.sort(key=lambda x: x['left'])
                    rows.append(current_row)
                    current_row = [item]
        
        if current_row:
            current_row.sort(key=lambda x: x['left'])
            rows.append(current_row)
        
        # Reconstruct with proper spacing
        final_text = ""
        for row in rows:
            row_text = ""
            prev_right = 0
            
            for item in row:
                left = item['left']
                
                # Add spacing
                if prev_right > 0:
                    gap = left - prev_right
                    if gap > 50:
                        row_text += "\t\t"
                    elif gap > 20:
                        row_text += "\t"
                    else:
                        row_text += " "
                
                row_text += item['text']
                prev_right = left + item['width']
            
            final_text += row_text.strip() + "\n"
        
        return final_text
    
    def select_best_extraction(self, extractions: List[Tuple[str, str]]) -> str:
        """
        Select the best text extraction based on quality metrics.
        """
        if not extractions:
            return ""
        
        if len(extractions) == 1:
            return extractions[0][1]
        
        scored_extractions = []
        
        for name, text in extractions:
            score = self.calculate_text_quality_score(text)
            scored_extractions.append((score, name, text))
            self.log(f"{name}: quality score = {score:.2f}")
        
        # Sort by score (higher is better)
        scored_extractions.sort(reverse=True)
        
        best_score, best_name, best_text = scored_extractions[0]
        self.log(f"Selected: {best_name} (score: {best_score:.2f})")
        
        return best_text
    
    def calculate_text_quality_score(self, text: str) -> float:
        """
        Calculate quality score for extracted text based on invoice-specific criteria.
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 0.0
        
        # Length bonus (more text is generally better)
        score += min(len(text) / 1000, 1.0) * 20
        
        # Currency symbols presence
        currency_count = len(re.findall(r'[\$€£¥₹]', text))
        score += min(currency_count * 5, 25)
        
        # Number presence (invoices have many numbers)
        number_count = len(re.findall(r'\d+', text))
        score += min(number_count * 2, 30)
        
        # Common invoice keywords
        invoice_keywords = ['total', 'subtotal', 'tax', 'amount', 'date', 'invoice', 'bill']
        keyword_count = sum(1 for keyword in invoice_keywords if keyword.lower() in text.lower())
        score += keyword_count * 5
        
        # Penalize excessive special characters (OCR artifacts)
        special_char_ratio = len(re.findall(r'[^\w\s\$.,%-]', text)) / max(len(text), 1)
        score -= special_char_ratio * 50
        
        # Penalize very short lines (fragmented text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            if avg_line_length < 5:
                score -= 20
        
        return max(score, 0.0)
    
    def reconstruct_lines_from_ocr(self, ocr_results: List) -> str:
        """
        Reconstruct text lines from OCR bounding box results with improved accuracy.
        """
        if not ocr_results:
            return ""
        
        # Calculate dynamic threshold based on text heights
        heights = []
        for bbox, text, conf in ocr_results:
            if isinstance(bbox, list) and len(bbox) >= 4:
                h = abs(bbox[2][1] - bbox[0][1])
                heights.append(h)
        
        if not heights:
            return " ".join([result[1] for result in ocr_results])
        
        median_height = sorted(heights)[len(heights)//2]
        y_threshold = median_height * 0.6  # More generous threshold for blurry images
        
        # Sort by Y coordinate (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1] if isinstance(x[0], list) else 0)
        
        lines = []
        current_line = []
        
        for item in sorted_results:
            bbox, text, conf = item
            
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
                
            y_top = bbox[0][1]
            
            if not current_line:
                current_line.append(item)
                continue
            
            # Calculate average Y of current line
            avg_y = sum(x[0][0][1] for x in current_line if isinstance(x[0], list)) / len(current_line)
            
            # Check if this item belongs to the current line
            if abs(y_top - avg_y) < y_threshold:
                current_line.append(item)
            else:
                # Finish current line and start new one
                if current_line:
                    lines.append(current_line)
                current_line = [item]
        
        if current_line:
            lines.append(current_line)
        
        # Process each line: sort by X coordinate and join
        final_text = ""
        for line in lines:
            # Sort items in line by X coordinate (left to right)
            line.sort(key=lambda x: x[0][0][0] if isinstance(x[0], list) else 0)
            
            # Join text with appropriate spacing
            line_texts = []
            prev_x_end = 0
            
            for item in line:
                bbox, text, conf = item
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x_start = bbox[0][0]
                    
                    # Add extra space if there's a significant gap
                    if prev_x_end > 0 and x_start - prev_x_end > median_height:
                        line_texts.append("   ")  # Tab-like spacing
                    
                    line_texts.append(text)
                    prev_x_end = bbox[1][0]  # Right edge of current text
                else:
                    line_texts.append(text)
            
            final_text += " ".join(line_texts) + "\n"
        
        return final_text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
        
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    
    def clean_and_enhance_text(self, text: str) -> str:
        """
        Advanced text cleaning and enhancement for better LLM processing.
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Extract all amounts to identify outliers
        all_amounts = re.findall(r'[\$€£¥₹]?([0-9,]+\.?[0-9]{0,2})', text)
        numeric_amounts = []
        for amount in all_amounts:
            try:
                numeric_amounts.append(float(amount.replace(',', '')))
            except:
                continue
        
        # Calculate reasonable amount range
        if numeric_amounts:
            numeric_amounts.sort()
            median = numeric_amounts[len(numeric_amounts)//2] if numeric_amounts else 0
            outlier_threshold = median * 10 if median > 0 else 10000
        else:
            outlier_threshold = 10000
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Fix common OCR errors
            line = self.fix_common_ocr_errors(line)
            
            # Skip obviously corrupted lines
            if self.is_corrupted_line(line):
                continue
            
            # Skip lines with unreasonable amounts
            amounts_in_line = re.findall(r'[\$€£¥₹]([0-9,]+\.?[0-9]{0,2})', line)
            has_reasonable_amount = True
            for amount in amounts_in_line:
                try:
                    val = float(amount.replace(',', ''))
                    if val > outlier_threshold:
                        has_reasonable_amount = False
                        break
                except:
                    continue
            
            if not has_reasonable_amount:
                self.log(f"Skipping line with unreasonable amount: {line}")
                continue
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        self.log(f"Text cleaning: {len(lines)} -> {len(cleaned_lines)} lines")
        
        return cleaned_text
    
    def fix_common_ocr_errors(self, line: str) -> str:
        """Fix common OCR errors in text"""
        # Currency symbol fixes
        line = re.sub(r'\b8([0-9]{1,3}[,.][0-9]{3}[,.][0-9]{2})', r'$\1', line)  # 82,500.00 -> $2,500.00
        line = re.sub(r'\b8([0-9]{1,3}[,.][0-9]{2})', r'$\1', line)              # 82.50 -> $2.50
        line = re.sub(r'\bS([0-9])', r'$\1', line)                               # S500 -> $500
        line = re.sub(r'\b0([0-9])', r'$\1', line)                               # 0500 -> $500 (sometimes)
        
        # Common character substitutions
        line = line.replace('|', 'I')  # Pipe to I
        line = line.replace('1l', 'll')  # 1l to ll
        line = line.replace('0O', 'OO')  # 0O to OO
        
        # Clean up spacing
        line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single
        line = re.sub(r'\s*\$\s*', '$', line)  # Clean dollar sign spacing
        
        return line.strip()
    
    def is_corrupted_line(self, line: str) -> bool:
        """Check if a line appears to be corrupted OCR output"""
        if len(line) < 2:
            return True
        
        # Too many special characters
        special_char_ratio = len(re.findall(r'[^\w\s\$.,%-]', line)) / len(line)
        if special_char_ratio > 0.4:
            return True
        
        # Too many single characters separated by spaces
        single_chars = len(re.findall(r'\b\w\b', line))
        if single_chars > len(line) * 0.3:
            return True
        
        return False
    
    def process_with_llm(self, invoice_text: str, model_type: str = "ollama") -> Optional[Dict]:
        """
        Process extracted text with LLM to extract structured invoice data.
        Supports multiple LLM backends.
        """
        if not invoice_text or len(invoice_text.strip()) < 10:
            self.log("Insufficient text for LLM processing")
            return None
        
        # Clean and enhance text before LLM processing
        cleaned_text = self.clean_and_enhance_text(invoice_text)
        
        if model_type.lower() == "ollama":
            return self.process_with_ollama(cleaned_text)
        elif model_type.lower() == "lmstudio":
            return self.process_with_lmstudio(cleaned_text)
        else:
            self.log(f"Unsupported model type: {model_type}")
            return None
    
    def process_with_ollama(self, invoice_text: str) -> Optional[Dict]:
        """Process with Ollama API"""
        prompt = self.create_extraction_prompt(invoice_text)
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1500,
                "seed": random.randint(0, 100000)
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get('response', '')
            
            return self.parse_llm_response(raw_response)
            
        except requests.exceptions.RequestException as e:
            self.log(f"Ollama API error: {e}")
            return None
    
    def process_with_lmstudio(self, invoice_text: str) -> Optional[Dict]:
        """Process with LM Studio API"""
        prompt = self.create_extraction_prompt(invoice_text)
        
        url = "http://localhost:1234/v1/chat/completions"
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result['choices'][0]['message']['content']
            
            return self.parse_llm_response(raw_response)
            
        except requests.exceptions.RequestException as e:
            self.log(f"LM Studio API error: {e}")
            return None
    
    def create_extraction_prompt(self, invoice_text: str) -> str:
        """Create optimized prompt for invoice data extraction"""
        return f"""Extract invoice data as JSON. Look carefully for:

1. VENDOR NAME: First company name at top
2. DATE: Any date format (MM/DD/YYYY, DD-MM-YYYY, etc.)
3. LINE ITEMS: Each service/product with price
4. QUANTITIES: Numbers before descriptions or in separate columns
5. UNIT PRICES: Price per item/service
6. AMOUNTS: Total for each line (quantity × unit_price)
7. SUBTOTAL: Sum of all line items
8. TAX: Tax amount
9. TOTAL: Final amount

IMPORTANT:
- Look for table structures with QTY, DESCRIPTION, UNIT PRICE, AMOUNT columns
- If you see "2 × $50.00 = $100.00", extract: quantity=2, unit_price=50.00, amount=100.00
- If only total amount shown, set quantity=1, unit_price=amount
- Extract EXACT subtotal from invoice (don't calculate)
- Handle OCR errors: "8" might be "$", "S" might be "$"

Output valid JSON only:
{{
  "vendor_name": null,
  "date": null,
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "total": null,
  "items": [
    {{
      "item_name": "description",
      "unit_price": 0.00,
      "quantity": 1,
      "amount": 0.00
    }}
  ]
}}

INVOICE TEXT:
{invoice_text}

JSON:"""
    
    def parse_llm_response(self, raw_response: str) -> Optional[Dict]:
        """Parse LLM response and extract JSON"""
        if not raw_response:
            return None
        
        # Find JSON in response
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}') + 1
        
        if start_idx == -1:
            self.log("No JSON found in LLM response")
            return None
        
        json_str = raw_response[start_idx:end_idx] if end_idx > start_idx else raw_response[start_idx:]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed_json = self.fix_json_string(json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                self.log(f"JSON parsing failed: {e}")
                self.log(f"Raw JSON: {json_str[:200]}...")
                return None
    
    def fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Balance brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
    def extract_invoice_data(self, file_path: str, model_type: str = "ollama") -> Optional[Dict]:
        """
        Main function to extract invoice data from image or PDF.
        
        Args:
            file_path: Path to invoice file
            model_type: LLM backend to use ("ollama" or "lmstudio")
        
        Returns:
            Structured invoice data or None if extraction fails
        """
        self.log(f"Processing invoice: {file_path}")
        
        try:
            # Step 1: Extract text using ensemble OCR
            self.log("Step 1: Extracting text with ensemble OCR...")
            extracted_text = self.extract_text_ensemble(file_path)
            
            if not extracted_text or len(extracted_text.strip()) < 20:
                self.log("OCR extraction failed or insufficient text")
                return None
            
            self.log(f"OCR extracted {len(extracted_text)} characters")
            
            # Step 2: Process with LLM
            self.log(f"Step 2: Processing with {model_type.upper()}...")
            result = self.process_with_llm(extracted_text, model_type)
            
            if result:
                self.log("LLM processing successful")
                return result
            else:
                self.log("LLM processing failed")
                return None
                
        except Exception as e:
            self.log(f"Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_debug_images(self, processed_images: List[Tuple[str, np.ndarray]], output_dir: str = "debug_images"):
        """Save processed images for debugging"""
        if not self.debug:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, img in processed_images:
            output_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(output_path, img)
            self.log(f"Saved debug image: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Invoice OCR with LLM Processing')
    parser.add_argument('file_path', nargs='?', help='Path to invoice image or PDF')
    parser.add_argument('--model', choices=['ollama', 'lmstudio'], default='ollama', 
                       help='LLM backend to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Initialize enhanced OCR system
    ocr = EnhancedInvoiceOCR(debug=args.debug)
    
    # Determine file path
    if args.file_path:
        invoice_path = args.file_path
    else:
        # Scan current directory for invoice files
        potential_files = [f for f in os.listdir('.') 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp'))]
        
        if not potential_files:
            print("No image/PDF files found in current directory.")
            print("Usage: python enhanced_invoice_ocr.py <path_to_invoice>")
            return
        
        print("Available files:")
        for i, f in enumerate(potential_files):
            print(f"{i+1}. {f}")
        
        try:
            selection = input("\nEnter number to process (or 'q' to quit): ")
            if selection.lower() == 'q':
                return
            idx = int(selection) - 1
            if 0 <= idx < len(potential_files):
                invoice_path = potential_files[idx]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
    
    # Process invoice
    try:
        result = ocr.extract_invoice_data(invoice_path, args.model)
        
        if result:
            print("\n" + "="*60)
            print("EXTRACTED INVOICE DATA")
            print("="*60)
            print(json.dumps(result, indent=2))
            print("="*60)
            
            # Save result
            output_name = f"{os.path.splitext(os.path.basename(invoice_path))[0]}_extracted.json"
            with open(output_name, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {output_name}")
            
            # Display formatted summary
            print("\nFORMATTED SUMMARY:")
            print("-" * 40)
            print(f"Vendor: {result.get('vendor_name', 'Not found')}")
            print(f"Date: {result.get('date', 'Not found')}")
            print(f"Items: {len(result.get('items', []))}")
            print(f"Subtotal: ${result.get('subtotal', 0)}")
            print(f"Tax: ${result.get('tax_amount', 0)}")
            print(f"Discount: ${result.get('discount_amount', 0)}")
            print(f"Total: ${result.get('total', 0)}")
            
        else:
            print("Failed to extract invoice data")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()