import cv2
import re
import pytesseract
from PIL import Image
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import os



import requests
import json
import random

class FocusedInvoiceOCR:
    """
    Simplified OCR system that extracts only specific invoice fields:
    - Company name
    - Date
    - Table items (description, quantity, price)
    - Final total
    """
   
    def __init__(self):
        pass
    
    def preprocess_image(self, image_path):
        """Advanced image preprocessing for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize image if too large
        height, width = img.shape[:2]
        if width > 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Apply different thresholding techniques and combine
        # Method 1: Adaptive threshold
        thresh1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 2: Otsu's threshold
        _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both thresholding methods
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
        
        return final
    
    def extract_text(self, file_path):
        """Extract text from image or PDF"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        else:
            return self.extract_text_from_image(file_path)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text using EasyOCR with layout preservation"""
        import easyocr
        import numpy as np
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        try:
            # Read text from image (raw image for best quality)
            results = reader.readtext(image_path)
            
            # Reconstruct text preserving layout (lines)
            # Filter low confidence first
            high_conf_results = [r for r in results if r[2] > 0.3]
            
            extracted_text = self.reconstruct_lines(high_conf_results)
            
            print(f"EasyOCR extracted {len(extracted_text)} characters")
            return extracted_text
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def extract_company_name(self, text):
        """Extract company name with enhanced patterns"""
        lines = text.split('\n')
        
        # Enhanced company name patterns
        company_patterns = [
            r'([A-Z][A-Za-z\s&.,]+(?:Inc|LLC|Ltd|Corp|Company|Co\.|Corporation|LTD|INC))',
            r'([A-Z][A-Za-z\s&.,]{5,50})',  # Capitalized text
            r'^\s*([A-Za-z][A-Za-z\s&.,]{10,60})\s*$',  # Long text lines
            r'([A-Z\s]{3,30})',  # All caps company names
        ]
        
       
                
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1).strip()
                if len(company) > 3:
                    return company
        
        # Fallback: return first substantial non-numeric line
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 5 and not re.search(r'^\d+', line) and not any(word in line.lower() for word in ['invoice', 'bill', 'date']):
                return line
        
        return "Not found"
    
    def extract_date(self, text):
        """Extract invoice date"""
        date_match = re.search(r'Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})', text)
        if date_match:
            date_str = date_match.group(1)
            # Convert to ISO format
            parts = date_str.split('/')
            if len(parts) == 3:
                month, day, year = parts
                if len(year) == 2:
                    year = '20' + year
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None
    
    def extract_subtotal(self, text):
        """Extract subtotal amount"""
        subtotal_patterns = [
            r'(?:Subtotal|SUBTOTAL|Sub\s*Total|Sub-Total)\s*[:]?\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?:^|\n)\s*Subtotal\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)(?:^|\s)subtotal\s*[:]?\s*[\$€£₹]?\s*([\d,]+\.\d{2})'
        ]
        
        for pattern in subtotal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        return "0"
    
    def extract_tax(self, text):
        """Extract tax amount with enhanced patterns"""
        tax_patterns = [
            r'(?i)(?:tax|TAX|GST|VAT|Sales\s*Tax|HST)\s*[:]?\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)(?:tax|GST|VAT)\s*\([\d.%]+\)\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)sales\s*tax\s*\([\d.%]+\)\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)(?:^|\n)\s*tax\s*[\$€£]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)\b(?:tax|gst|vat)\b.*?([\d,]+\.\d{2})',
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '')
        return "0"
    
    def extract_discount(self, text):
        """Extract discount amount with comprehensive patterns for PDF"""
        # Print text for debugging
        print("\n=== SEARCHING FOR DISCOUNT ===")
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'discount' in line.lower():
                print(f"Line {i}: {line}")
        
        discount_patterns = [
            r'(?i)discount\s*[:]?\s*[\$€£₹]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)disc\s*[:]?\s*[\$€£₹]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)discount\s*\([\d.%]+\)\s*[\$€£₹]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)discount\s*-\s*[\$€£₹]?\s*([\d,]+\.?\d{0,2})',
            r'(?i)discount\s*\([\$€£₹]?([\d,]+\.?\d{0,2})\)',
            r'(?i)\bdiscount\b.*?([\d,]+\.\d{2})',
            r'(?i)\bdiscount\b.*?([\d,]+)',
            r'(?i)less\s*discount\s*[:]?\s*[\$€£₹]?\s*([\d,]+\.?\d{0,2})',
        ]
        
        for pattern in discount_patterns:
            match = re.search(pattern, text)
            if match:
                discount_value = match.group(1).replace(',', '')
                print(f"Found discount: {discount_value} using pattern: {pattern}")
                return discount_value
        
        print("No discount found")
        return "0"
    
    def extract_total_amount(self, text):
        """Extract final total amount with enhanced detection"""
        # Look for specific amounts in the text
        amounts = re.findall(r'\$([\d,]+(?:\.\d{2})?)', text)
        if amounts:
            # Convert to numbers and find the largest (likely the total)
            numeric_amounts = []
            for amount in amounts:
                try:
                    numeric_amounts.append(float(amount.replace(',', '')))
                except:
                    continue
            if numeric_amounts:
                return str(max(numeric_amounts))
        return "0"
    
    def extract_table_data(self, text):
        """Extract table items with enhanced detection"""
        lines = text.split('\n')
        table_data = []
        
        # Based on the actual text structure, look for vehicle entries with prices
        vehicle_lines = []
        price_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Look for vehicle descriptions
            if any(word in line.lower() for word in ['honda', 'chevy', 'suburban', 'crv']):
                vehicle_lines.append((i, line))
            
            # Look for standalone price lines
            if re.match(r'^\$\d+$', line):
                price_lines.append((i, line.replace('$', '')))
        
        # Match vehicles with their prices (prices usually come after vehicle descriptions)
        for v_idx, vehicle in vehicle_lines:
            # Find the next price after this vehicle
            for p_idx, price in price_lines:
                if p_idx > v_idx and p_idx - v_idx <= 5:  # Price within 5 lines after vehicle
                    table_data.append({
                        'description': vehicle,
                        'quantity': '1',
                        'unit_price': price,
                        'amount': price
                    })
                    break
        
        print(f"\nTotal items found: {len(table_data)}")
        return table_data
    
    def parse_line_simple(self, line):
        """Enhanced line parsing for table data"""
        line = line.strip()
        if not line or len(line) < 3:
            return None
        
        # Find all numbers (including decimals)
        numbers = re.findall(r'\d+(?:\.\d{1,2})?', line)
        if not numbers:
            return None
        
        # Extract description - everything before the first number or specific patterns
        desc_match = re.match(r'^([A-Za-z][^\d]*?)(?=\d|$)', line)
        if desc_match:
            description = desc_match.group(1).strip()
        else:
            # Fallback: remove all numbers and clean
            description = line
            for num in numbers:
                description = description.replace(num, ' ', 1)
            description = re.sub(r'[^\w\s-]', ' ', description).strip()
        
        description = ' '.join(description.split())
        
        if not description or len(description) < 2:
            return None
        
        # Initialize result with defaults
        result = {
            'description': description,
            'quantity': '1',
            'unit_price': '0',
            'amount': '0'
        }
        
        # Smart number mapping based on count and patterns
        if len(numbers) >= 3:
            # Likely: qty, unit_price, amount
            result['quantity'] = numbers[0]
            result['unit_price'] = numbers[-2]
            result['amount'] = numbers[-1]
        elif len(numbers) == 2:
            # Likely: unit_price, amount (qty assumed 1)
            result['unit_price'] = numbers[0]
            result['amount'] = numbers[1]
        elif len(numbers) == 1:
            # Single number - could be amount or price
            result['amount'] = numbers[0]
            result['unit_price'] = numbers[0]
        
        # Look for explicit quantity indicators
        qty_patterns = [
            r'(?i)(?:qty|quantity|q)\s*[:]?\s*(\d+)',
            r'(\d+)\s*(?:x|X|pcs|PCS|units|each)',
            r'(?:x|X)\s*(\d+)'
        ]
        
        for pattern in qty_patterns:
            match = re.search(pattern, line)
            if match:
                result['quantity'] = match.group(1)
                break
        
        return result
    
    def is_header_line(self, line_lower):
        """Check if a line contains table headers"""
        header_keywords = [
            'description', 'item', 'product', 'service', 'desc',
            'quantity', 'qty', 'qnty', 'units', 'amount',
            'price', 'rate', 'cost', 'total', 'value'
        ]
        
        found_count = sum(1 for keyword in header_keywords if keyword in line_lower)
        return found_count >= 2
    
    def map_column_positions(self, header_line):
        """Map column positions based on header text"""
        positions = {}
        header_lower = header_line.lower()
        
        # Define header patterns and their positions
        header_patterns = {
            'description': r'(?:item|description|desc|product|service)',
            'quantity': r'(?:qty|quantity|qnty|units)',
            'unit_price': r'(?:price|rate|unit\s*price|unit\s*rate|cost)',
            'amount': r'(?:amount|total|subtotal|value)'
        }
        
        for field, pattern in header_patterns.items():
            match = re.search(pattern, header_lower)
            if match:
                positions[field] = match.start()
        
        return positions
    
    def parse_table_line_by_headers(self, line, column_positions):
        """Parse table line using column header positions"""
        line = line.strip()
        if not line or not column_positions:
            return None
        
        # Extract all numbers from the line
        numbers = re.findall(r'\d+\.?\d{0,2}', line)
        if not numbers:
            return None
        
        # Initialize result
        result = {
            'description': '',
            'quantity': '1',
            'unit_price': '0',
            'amount': '0'
        }
        
        # Sort columns by position
        sorted_columns = sorted(column_positions.items(), key=lambda x: x[1])
        
        # Extract description (text before first number or in description column area)
        desc_pos = column_positions.get('description', 0)
        desc_end = min([pos for field, pos in column_positions.items() if field != 'description' and pos > desc_pos] + [len(line)])
        
        description_text = line[desc_pos:desc_end].strip()
        # Clean description by removing numbers
        for num in numbers:
            description_text = description_text.replace(num, ' ', 1)
        result['description'] = re.sub(r'[^\w\s]', ' ', description_text).strip()
        result['description'] = ' '.join(result['description'].split())
        
        # Map numbers to columns based on positions and context
        if len(numbers) >= 3:
            # Standard case: quantity, unit_price, amount
            result['quantity'] = numbers[0]
            result['unit_price'] = numbers[-2]
            result['amount'] = numbers[-1]
        elif len(numbers) == 2:
            # Two numbers: likely unit_price and amount
            result['unit_price'] = numbers[0]
            result['amount'] = numbers[1]
        elif len(numbers) == 1:
            # Single number: could be price or amount
            result['amount'] = numbers[0]
            result['unit_price'] = numbers[0]
        
        # Try to find quantity in text patterns
        qty_patterns = [
            r'(?:quantity|qty|QTY)\s*[:]?\s*(\d+)',
            r'(\d+)\s*(?:x|X|pcs|PCS|units)',
            r'(?:x|X)\s*(\d+)'
        ]
        
        for pattern in qty_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result['quantity'] = match.group(1)
                break
        
        return result if result['description'] else None
    
    def process_with_lm_studio(self, invoice_text, table_rows=""):
        """Process extracted OCR text using LM Studio with gpt-oss-20b"""
        
        prompt = self.extract_invoice_data_with_prompt(invoice_text, table_rows)
        
        # LM Studio API endpoint (OpenAI-compatible)
        url = "http://localhost:1234/v1/chat/completions"
        
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result['choices'][0]['message']['content']
            
            # Extract JSON from response
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = raw_response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                print("No valid JSON found in response")
                print(f"Raw response: {raw_response}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {raw_response}")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"LM Studio API error: {e}")
            return None
    
    def reconstruct_lines(self, ocr_results):
        """
        Reconstruct lines from OCR results by grouping text blocks that are vertically close.
        Uses dynamic threshold based on text height.
        """
        if not ocr_results:
            return ""

        # Calculate median text height for dynamic threshold
        heights = []
        for bbox, _, _ in ocr_results:
            h = bbox[2][1] - bbox[0][1]
            heights.append(h)
        
        median_height = sorted(heights)[len(heights)//2] if heights else 20
        y_threshold = median_height * 0.5
        
        # Sort by top-left Y coordinate
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
        
        lines = []
        current_line = []
        
        for item in sorted_results:
            bbox, text, conf = item
            y_top = bbox[0][1]
            
            if not current_line:
                current_line.append(item)
                continue
                
            # Calculate average Y of the current line
            avg_y = sum(x[0][0][1] for x in current_line) / len(current_line)
            
            # If the new item is within threshold of the line's average Y
            if abs(y_top - avg_y) < y_threshold:
                current_line.append(item)
            else:
                # Finish current line and start new one
                lines.append(current_line)
                current_line = [item]
        
        if current_line:
            lines.append(current_line)
            
        # Process each line: sort by X and join
        final_text = ""
        for line in lines:
            # Sort items in line by X coordinate
            line.sort(key=lambda x: x[0][0][0])
            # Join with tabs or logical spaces
            line_text = "   ".join([x[1] for x in line])
            final_text += line_text + "\n"
            
        return final_text

    def clean_ocr_text(self, text):
        """Dynamic OCR text cleaning without hardcoded values"""
        lines = text.split('\n')
        cleaned_lines = []
        
        # Find potential vendor name (first meaningful line)
        vendor_line = None
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and not line.isdigit() and not re.match(r'^[\d\s\-\.]+$', line):
                vendor_line = line
                break
        
        # Extract all dollar amounts to identify outliers
        all_amounts = re.findall(r'\$([\d,]+\.?\d{0,2})', text)
        numeric_amounts = []
        for amount in all_amounts:
            try:
                numeric_amounts.append(float(amount.replace(',', '')))
            except:
                continue
        
        # Calculate median to identify outliers
        if numeric_amounts:
            numeric_amounts.sort()
            median = numeric_amounts[len(numeric_amounts)//2]
            # Consider amounts > 3x median as potential OCR errors
            outlier_threshold = median * 3 if median > 0 else 1000
        else:
            outlier_threshold = 1000
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Fix common OCR HTML entities
            line = re.sub(r'&#\d+;', '', line)
            line = line.replace('&gt;', '>').replace('&lt;', '<').replace('&quot;', '"')
            
            # Clean up spacing around dollar signs
            line = re.sub(r'\s*\$\s*', '$', line)
            
            keep_line = False
            
            # Keep vendor line
            if vendor_line and line == vendor_line:
                keep_line = True
            
            # Keep lines with reasonable dollar amounts
            elif '$' in line:
                amounts_in_line = re.findall(r'\$([\d,]+\.?\d{0,2})', line)
                has_reasonable_amount = False
                for amount in amounts_in_line:
                    try:
                        val = float(amount.replace(',', ''))
                        if val <= outlier_threshold:  # Not an outlier
                            has_reasonable_amount = True
                            break
                    except:
                        continue
                
                if has_reasonable_amount:
                    keep_line = True
            
            # Keep service/item description lines
            elif any(keyword in line.lower() for keyword in [
                'service', 'initial', 'subtotal', 'total', 'tax', 'discount'
            ]):
                keep_line = True
            
            # Keep date lines
            elif re.search(r'\d{1,2}/\d{1,2}/\d{4}', line):
                keep_line = True
            
            # Skip obviously garbled lines (too many special characters)
            special_char_ratio = len(re.findall(r'[^\w\s\$\.,\-]', line)) / max(len(line), 1)
            if special_char_ratio > 0.3:  # More than 30% special characters
                keep_line = False
            
            if keep_line:
                # Late-stage clean up: Fix common OCR currency errors
                # invalid: 82,500.00 (likely $2,500.00 if it's a huge outlier)
                # pattern: 8 followed by digit, comma, digit
                
                # specific fix for "8" -> "$" at start of amounts
                # Regex for "8" at start of word followed by digits and comma/dot
                # e.g. 82,500.00 -> $2,500.00
                # Old regex was too strict. New one handles thousands group:
                # 8 followed by 1-3 digits, then a comma/dot, then more digits
                line = re.sub(r'\b8(\d{1,3}[,.]\d{3}[,.]\d{2})', r'$\1', line) # Matches 82,500.00
                line = re.sub(r'\b8(\d{1,3}[,.]\d{2})', r'$\1', line)          # Matches 82.50
                line = re.sub(r'\b8(\d{1,3}[,.]\d{3})', r'$\1', line)          # Matches 82,000 (no cents)

                # Fix S -> $
                line = re.sub(r'\bS(\d)', r'$\1', line)
                
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_table_headers(self, text):
        """Remove obvious table headers from OCR text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that look like table headers
            # (short lines with common header words)
            if len(line) <= 5 and line.upper() in ['TEM', 'ITEM', 'QTY', 'COST', 'PRICE', 'DESC']:
                continue
                
            # Skip lines that are just column separators or formatting
            if all(c in '-_=|+' for c in line.replace(' ', '')):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_invoice_data_hybrid(self, file_path):
        """Two-step process: OCR extraction + Ollama processing"""
        print(f"Processing: {file_path}")
        
        # Step 1: Extract text using OCR
        print("Step 1: Extracting text with OCR...")
        extracted_text = self.extract_text(file_path)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            print("OCR extraction failed or insufficient text")
            return None
        
        print(f"OCR extracted {len(extracted_text)} characters")
        
        # Step 1.2: Fix common OCR artifacts (8 -> $, S -> $)
        # This must be done BEFORE sending to LLM
        
        # Fix 8 -> $ for patterns like 82,500.00 -> $2,500.00
        extracted_text = re.sub(r'\b8(\d{1,3}[,.]\d{3}[,.]\d{2})', r'$\1', extracted_text) 
        extracted_text = re.sub(r'\b8(\d{1,3}[,.]\d{2})', r'$\1', extracted_text)
        extracted_text = re.sub(r'\b8(\d{1,3}[,.]\d{3})', r'$\1', extracted_text)

        # Fix S -> $
        extracted_text = re.sub(r'\bS(\d)', r'$\1', extracted_text) # S500.00 -> $500.00
        
        # Step 1.5: Clean table headers
        cleaned_text = self.clean_table_headers(extracted_text)
        
        # Debug: Show the cleaned OCR text
        print("\nCleaned OCR text being sent to LLM:")
        print("-" * 50)
        print(cleaned_text)
        print("-" * 50)
        
        # Step 2: Process cleaned text with Ollama
        print("Step 2: Processing with Ollama...")
        ollama_result = self.process_with_ollama(cleaned_text)
        
        if ollama_result:
            print("Ollama processing successful")
            return ollama_result
        else:
            print("Ollama processing failed")
            return None
    
    
    def process_with_ollama(self, invoice_text, table_rows=""):
        """Process invoice text using Ollama with Gemma2 2B model"""
        
        # Get the extraction prompt
        prompt = self.extract_invoice_data_with_prompt(invoice_text, table_rows)
        
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent output
                "top_p": 0.9,
                "num_predict": 1000,
                "seed": random.randint(0, 100000)  # Break cache
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get('response', '')
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = raw_response.find('{')
                end_idx = raw_response.rfind('}') + 1
                
                if start_idx != -1:
                    if end_idx <= start_idx:
                        # If finding from right failed (likely truncated), take to end
                        json_str = raw_response[start_idx:]
                    else:
                        json_str = raw_response[start_idx:end_idx]
                    
                    # Attempt cleanup and parsing
                    try:
                        parsed_json = json.loads(json_str)
                        return parsed_json
                    except json.JSONDecodeError:
                        # Smarter fix: Balance brackets
                        def balance_json(s):
                            stack = []
                            is_escaped = False
                            in_string = False
                            for char in s:
                                if is_escaped:
                                    is_escaped = False
                                    continue
                                if char == '\\':
                                    is_escaped = True
                                    continue
                                if char == '"':
                                    in_string = not in_string
                                    continue
                                if not in_string:
                                    if char == '{':
                                        stack.append('}')
                                    elif char == '[':
                                        stack.append(']')
                                    elif char == '}' or char == ']':
                                        if stack and stack[-1] == char:
                                            stack.pop()
                            # Append missing closing brackets in reverse order
                            return s + "".join(reversed(stack))
                        
                        fixed_str = balance_json(json_str.strip())
                        try:
                            return json.loads(fixed_str)
                        except json.JSONDecodeError:
                            print(f"Smart JSON fix failed on: {json_str[:50]}...")
                            return None
                else:
                    print("No valid JSON found in response")
                    print(f"Raw response: {raw_response}")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {raw_response}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
    
    def extract_invoice_data_with_prompt(self, invoice_text, table_rows=""):
        """Convert invoice text to exact JSON format using the extraction prompt"""
        
        prompt = f"""You are an invoice extraction engine. Extract ONLY the main billable services.
        
        CRITICAL: 
        1. VENDOR NAME is usually at the very top. Look for:
           - Large text at the start (e.g. "GILLS TREE...", "RELIANT PEST...")
           - Email domains (e.g. "@gillstreeservice.com" -> "Gills Tree Service")
           - Logos text at the top left/right.
        2. Extract valid JSON only.
Do not include markdown blocks (```json).
Do not add comments.
Ensure all JSON keys and values are properly quoted and valid.
Do not add trailing commas.
Output must match this exact schema:

{{
  "vendor_name": null,
  "date": null,
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "total": null,
  "items": [
    {{ "item_name": null, "unit_price": null, "quantity": null, "amount": null }}
  ]
}}

STRICT RULES:
1) Extract EVERY service line that has a price. Do not skip any.
2) If you see multiple dollar amounts in the text, usually each one corresponds to an item.
    - Exception: If a line has "$2500 $2500", it is ONE item of $2500.
3) Use the text exactly as it appears for descriptions.
4) "Discount" is NOT an item. Put it in "discount_amount".
5) "Card Processing Fee" IS an item.
6) Trust the "Total" printed on the invoice for the 'total' field.
5) "Discount" is NOT an item. Put it in "discount_amount".
6) Trust the "Total" printed on the invoice for the 'total' field.
7) EXTRACT the "Subtotal" printed on the invoice. Do NOT calculate it yourself.
8) Ensure extracted amounts are numbers (e.g. 1500.00, not "1,500.00").

For items:
- Focus on main service lines that have clear dollar amounts
- Look for patterns like: [Service Description] followed by [Dollar Amount]
- Ignore material lists and supply details
- Only include services where the amount makes sense with the subtotal

For totals:
- Find "Subtotal" and extract that exact amount
- Find "Total" and extract that exact amount
- Verify that item amounts add up to subtotal

INVOICE TEXT (OCR extracted):
<<<
{invoice_text}
>>>

Notes on Layout & Extraction Rules:
1.  **Line Items**: Look for lines with specific service descriptions AND a price.
    -   Example: "Service Description... $120.00" -> Item: "Service Description", Amount: 120.00
    -   Example: "Another Service... $150.00" -> Item: "Another Service", Amount: 150.00
2.  **Totals/Subtotals are NOT Items**:
    -   "Subtotal $1000.00" is a SUMMARY, NOT a billable item. Do not list it in the 'items' array.
    -   "Total $1000.00" is a SUMMARY.
    -   The 'items' array should sum up to the Subtotal.
3.  **Parsing Help**:
    -   "1-24 S150.00" -> The 'S' is likely a '$'.
    -   "82,500.00" -> The '8' is likely a '$' if the item price seems wrong.
    -   "Service... $120.00" is an item.
    -   "Item... $100.00 $100.00" -> Extract ONCE (ignore duplicate column).
    -   "Discount $50.00" -> Extract "Discount" or "Credit" lines as 'discount_amount'.
    -   "Total $..." -> Extract the final total.
 
OPTIONAL TABLE ROWS (if you have extracted rows separately):
<<<
{table_rows}
>>>"""
        
        return prompt
        
        return prompt
        
        return prompt
    
    def extract_invoice_data(self, file_path):
        """Main function to extract focused invoice data from image or PDF"""
        print(f"Processing: {file_path}")
        
        # Extract text
        text = self.extract_text(file_path)
        
        # Extract specific fields
        result = {
            'company_name': self.extract_company_name(text),
            'date': self.extract_date(text),
            'items': self.extract_table_data(text),
            'subtotal': self.extract_subtotal(text),
            'tax': self.extract_tax(text),
            'discount': self.extract_discount(text),
            'total_amount': self.extract_total_amount(text)
        }

        return result
    
    def display_results(self, result):
        """Display extracted data in a clean format"""
        print("\n" + "="*50)
        print("EXTRACTED INVOICE DATA")
        print("="*50)
        
        print(f"Company Name: {result.get('company_name', 'Not found')}")
        print(f"Date: {result.get('date', 'Not found')}")
        
        print("\nITEMS:")
        print("-" * 80)
        items = result.get('items', [])
        
        if items:
            print(f"{'Description':<30} {'Qty':<5} {'Unit Price':<12} {'Amount':<10}")
            print("-" * 80)
            for item in items:
                desc = item['description'][:27] + "..." if len(item['description']) > 30 else item['description']
                print(f"{desc:<30} {item['quantity']:<5} ${item['unit_price']:<11} ${item['amount']:<10}")
        else:
            print("No items found")
        
        print("\n" + "="*50)
        print("FINANCIAL SUMMARY")
        print("="*50)
        
        subtotal = result.get('subtotal', '0')
        total_amount = result.get('total_amount', '0')
        
        # Only show subtotal if it's different from total
        if subtotal != total_amount and subtotal != '0':
            print(f"Subtotal: ${subtotal}")

        if subtotal == total_amount and subtotal !=0:
            print(f"Total: ${subtotal   }"   )
        
        print(f"Tax: ${result.get('tax', '0')}")
        print(f"Discount: ${result.get('discount', '0')}")
        print(f"Total Amount: ${total_amount}")
        print("\n" + "="*50)

# Usage example
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Invoice OCR Extraction')
    parser.add_argument('file_path', nargs='?', help='Path to invoice image or PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Initialize OCR system
    ocr = FocusedInvoiceOCR()
    
    # Determine file path
    if args.file_path:
        invoice_path = args.file_path
    else:
        # Fallback to scanning current directory for common image formats
        potential_files = [f for f in os.listdir('.') 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')) 
                          and f.lower() != "debug_layout.py"]
        
        if not potential_files:
            print("No image/PDF files provided or found in current directory.")
            print("Usage: python invoice_ocr.py <path_to_invoice>")
            return
            
        print("No file specified. Found the following potential files:")
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

    try:
        # Use the hybrid approach
        result = ocr.extract_invoice_data_hybrid(invoice_path)
        
        if result:
            print("\nHybrid Extraction Result:")
            print("=" * 50)
            print(json.dumps(result, indent=2))
            print("=" * 50)
            
            # Save result
            output_name = f"{os.path.splitext(os.path.basename(invoice_path))[0]}_data.json"
            with open(output_name, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to '{output_name}'")
            return result
        else:
            print("Failed to process invoice")
            return None
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()