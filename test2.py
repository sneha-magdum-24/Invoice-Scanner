import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

class MultiModelInvoiceOCR:
    """
    Multi-model OCR system that combines multiple OCR engines
    for robust invoice text extraction
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self, use_tesseract=True, use_easyocr=True, 
                          use_paddleocr=False, use_kraken=False):
        """
        Initialize selected OCR models
        """
        print("Initializing OCR models...")
        
        # Tesseract (Google)
        if use_tesseract:
            try:
                import pytesseract
                self.models['tesseract'] = pytesseract
                print("✓ Tesseract OCR initialized")
            except ImportError:
                print("✗ Tesseract not available. Install: pip install pytesseract")
        
        # EasyOCR (Preferred for invoices)
        if use_easyocr:
            try:
                import easyocr
                self.models['easyocr'] = easyocr.Reader(['en'], gpu=False)
                print("✓ EasyOCR initialized")
            except ImportError:
                print("✗ EasyOCR not available. Install: pip install easyocr")
        
        # PaddleOCR (Baidu - good for multilingual)
        if use_paddleocr:
            try:
                from paddleocr import PaddleOCR
                self.models['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en')
                print("✓ PaddleOCR initialized")
            except ImportError:
                print("✗ PaddleOCR not available. Install: pip install paddlepaddle paddleocr")
        
        # Kraken (Historical documents)
        if use_kraken:
            try:
                import kraken
                self.models['kraken'] = kraken
                print("✓ Kraken OCR initialized")
            except ImportError:
                print("✗ Kraken not available")
        
        print(f"Total models loaded: {len(self.models)}")
    
    def preprocess_image(self, image_path):
        """
        Enhanced image preprocessing for better OCR
        """
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
        else:
            # Assume it's already a numpy array
            img = image_path
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Multiple preprocessing techniques
        processed_images = {}
        
        # 1. Original grayscale
        processed_images['original_gray'] = gray
        
        # 2. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        processed_images['adaptive'] = adaptive
        
        # 3. Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images['otsu'] = otsu
        
        # 4. Denoised
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        processed_images['denoised'] = denoised
        
        # 5. Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images['sharpened'] = sharpened
        
        # 6. Resized (if too small)
        height, width = gray.shape
        if width < 800 or height < 800:
            scale = 800 / min(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(gray, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
            processed_images['resized'] = resized
        
        return processed_images
    
    def ocr_tesseract(self, image):
        """
        Extract text using Tesseract OCR
        """
        if 'tesseract' not in self.models:
            return None, 0
        
        try:
            # Try different configurations
            configs = [
                '--oem 3 --psm 3',  # Default
                '--oem 1 --psm 3',  # Legacy engine
                '--oem 3 --psm 6',  # Assume uniform block
                '--oem 3 --psm 11',  # Sparse text
            ]
            
            best_text = ""
            best_score = 0
            
            for config in configs:
                text = self.models['tesseract'].image_to_string(image, config=config)
                # Calculate confidence (simple heuristic)
                confidence = len(text.strip().split()) * 10
                if confidence > best_score:
                    best_text = text
                    best_score = confidence
            
            return best_text, best_score
        except Exception as e:
            print(f"Tesseract error: {e}")
            return None, 0
    
    def ocr_easyocr(self, image):
        """
        Extract text using EasyOCR
        """
        if 'easyocr' not in self.models:
            return None, 0
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            results = self.models['easyocr'].readtext(image_rgb, 
                                                      paragraph=True,
                                                      detail=0)
            
            # Combine results
            text = "\n".join(results)
            
            # Calculate confidence (average if details were available)
            confidence = len(text.strip().split()) * 15  # Heuristic
            
            return text, confidence
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None, 0
    
    def ocr_paddleocr(self, image):
        """
        Extract text using PaddleOCR
        """
        if 'paddleocr' not in self.models:
            return None, 0
        
        try:
            # Convert to BGR if grayscale
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Perform OCR
            result = self.models['paddleocr'].ocr(image_bgr, cls=True)
            
            # Extract text
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and line[1]:
                        text_lines.append(line[1][0])
            
            text = "\n".join(text_lines)
            confidence = len(text.strip().split()) * 12  # Heuristic
            
            return text, confidence
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return None, 0
    
    def extract_with_all_models(self, image_path):
        """
        Extract text using all available models
        """
        print(f"\nProcessing: {image_path}")
        
        # Preprocess image
        processed_images = self.preprocess_image(image_path)
        print(f"Generated {len(processed_images)} image variations")
        
        # Store results from all models
        all_results = {}
        
        # Test each preprocessing method with each model
        for prep_name, prep_image in processed_images.items():
            print(f"\n--- Using {prep_name} ---")
            
            # Convert to PIL Image for Tesseract if needed
            pil_image = Image.fromarray(prep_image)
            
            # Try Tesseract
            if 'tesseract' in self.models:
                text, confidence = self.ocr_tesseract(pil_image)
                if text:
                    key = f"tesseract_{prep_name}"
                    all_results[key] = {
                        'text': text,
                        'confidence': confidence,
                        'model': 'tesseract',
                        'preprocessing': prep_name
                    }
                    print(f"Tesseract: {len(text)} chars, confidence: {confidence}")
            
            # Try EasyOCR
            if 'easyocr' in self.models:
                text, confidence = self.ocr_easyocr(prep_image)
                if text:
                    key = f"easyocr_{prep_name}"
                    all_results[key] = {
                        'text': text,
                        'confidence': confidence,
                        'model': 'easyocr',
                        'preprocessing': prep_name
                    }
                    print(f"EasyOCR: {len(text)} chars, confidence: {confidence}")
            
            # Try PaddleOCR
            if 'paddleocr' in self.models:
                text, confidence = self.ocr_paddleocr(prep_image)
                if text:
                    key = f"paddleocr_{prep_name}"
                    all_results[key] = {
                        'text': text,
                        'confidence': confidence,
                        'model': 'paddleocr',
                        'preprocessing': prep_name
                    }
                    print(f"PaddleOCR: {len(text)} chars, confidence: {confidence}")
        
        return all_results
    
    def combine_results(self, all_results, method='confidence_weighted'):
        """
        Combine results from multiple OCR models
        """
        if not all_results:
            return None
        
        if method == 'best_confidence':
            # Select result with highest confidence
            best_key = max(all_results.keys(), 
                          key=lambda k: all_results[k]['confidence'])
            return all_results[best_key]['text']
        
        elif method == 'vote_based':
            # Simple voting system
            texts = [result['text'] for result in all_results.values()]
            
            # Find common lines
            all_lines = []
            for text in texts:
                lines = text.split('\n')
                all_lines.extend([line.strip() for line in lines if line.strip()])
            
            # Count occurrences
            from collections import Counter
            line_counts = Counter(all_lines)
            
            # Keep lines that appear in at least 2 results
            combined_lines = [line for line, count in line_counts.items() 
                             if count >= 2]
            
            return "\n".join(combined_lines)
        
        elif method == 'confidence_weighted':
            # Weighted combination based on confidence
            weighted_text = {}
            
            for key, result in all_results.items():
                text = result['text']
                confidence = result['confidence']
                
                # Split into lines and words
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        # Add weight to each word
                        words = line.split()
                        for word in words:
                            word_lower = word.lower()
                            if word_lower in weighted_text:
                                weighted_text[word_lower]['count'] += 1
                                weighted_text[word_lower]['confidence'] += confidence
                            else:
                                weighted_text[word_lower] = {
                                    'word': word,
                                    'count': 1,
                                    'confidence': confidence
                                }
            
            # Reconstruct text (simplified - in reality would need context)
            sorted_words = sorted(weighted_text.items(), 
                                 key=lambda x: x[1]['confidence'], 
                                 reverse=True)
            
            # Just return the best individual result for now
            best_key = max(all_results.keys(), 
                          key=lambda k: all_results[k]['confidence'])
            return all_results[best_key]['text']
    
    def extract_invoice_data(self, image_path, combine_method='best_confidence'):
        """
        Main function to extract invoice data
        """
        # Extract text with all models
        all_results = self.extract_with_all_models(image_path)
        
        if not all_results:
            return {
                'success': False,
                'error': 'No OCR models succeeded'
            }
        
        # Combine results
        combined_text = self.combine_results(all_results, combine_method)
        
        # Extract structured data
        extracted_data = self.parse_invoice_text(combined_text)
        
        # Store results
        self.results[image_path] = {
            'extraction_time': datetime.now().isoformat(),
            'all_results': all_results,
            'combined_text': combined_text,
            'extracted_data': extracted_data
        }
        
        return extracted_data
    
    def parse_invoice_text(self, text):
        """
        Parse extracted text into structured invoice data
        """
        # Common patterns for invoice extraction
        patterns = {
            'invoice_number': [
                r'(?:Invoice|Bill|INVOICE)\s*(?:No\.?|Number|#)?\s*[:]?\s*([A-Z0-9\-\s]+)',
                r'INV[-_]?(\d+)',
                r'Invoice\s+(\d+)',
                r'#\s*(\d+)'
            ],
            'invoice_date': [
                
                r'(?:Date|Invoice\s*Date|Dated|Issued|Date\s*of\s*Issue)\s*[:]?\s*([\d\/\-\.\sA-Za-z]+)',
                r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
            ],
            'due_date': [
                r'(?:Due\s*Date|Due|Pay\s*by|Payment\s*Due)\s*[:]?\s*([\d\/\-\.\sA-Za-z]+)',
                r'Due\s*[:]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
            ],
            'total_amount': [
                r'(?:Total|TOTAL|Amount\s*Due|Grand\s*Total|Balance)\s*[:]?\s*[\$€£]?\s*([\d,]+\.\d{2})',
                r'TOTAL\s*[\$€£]?\s*([\d,]+\.\d{2})',
                r'Amount\s*Due\s*[\$€£]?\s*([\d,]+\.\d{2})'
            ],
            'subtotal': [
                r'(?:Sub\s*Total|Subtotal|SUB\s*TOTAL)\s*[:]?\s*[\$€£]?\s*([\d,]+\.\d{2})',
                r'Subtotal\s*[\$€£]?\s*([\d,]+\.\d{2})'
            ],
            'tax_amount': [
                r'(?:Tax|TAX|VAT|GST)\s*(?:\(\d+%\))?\s*[:]?\s*[\$€£]?\s*([\d,]+\.\d{2})',
                r'Tax\s*[\$€£]?\s*([\d,]+\.\d{2})'
            ]
        }
        
        extracted = {}
        
        # Extract using patterns
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    extracted[field] = match.group(1).strip()
                    break
        
        # Extract all currency amounts
        currency_patterns = [
            r'[\$€£]\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*[\$€£]'
        ]
        
        amounts = []
        for pattern in currency_patterns:
            amounts.extend(re.findall(pattern, text))
        extracted['all_amounts'] = amounts
        
        # Try to find addresses (simple pattern)
        address_patterns = [
            r'(?:Bill\s*To|BILLED TO\s*TO|Invoice\s*To|Client)[:\s]*(.+?)(?:\n\s*\n|\n[A-Z])',
            r'(?:From|Vendor|Service\s*Provider)[:\s]*(.+?)(?:\n\s*\n|\n[A-Z])'
        ]
        
        addresses = []
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                addresses.append(match.group(1).strip())
        extracted['address_blocks'] = addresses
        
        # Extract table data (simplified)
        lines = text.split('\n')
        table_data = []
        in_table = False
        
        for line in lines:
            # Check if line looks like table header
            if any(word in line.lower() for word in ['item', 'description', 'qty', 'quantity', 'price', 'total', 'amount']):
                in_table = True
                continue
            
            if in_table:
                # Check if line has currency or numbers (simplified)
                if any(char in line for char in ['$', '€', '£']) or re.search(r'\d+\.\d{2}', line):
                    table_data.append(line.strip())
                elif line.strip() == '':
                    # End of table
                    in_table = False
        
        extracted['table_lines'] = table_data
        
        return extracted

# ============================================================================
# SIMPLIFIED VERSION (If you want something lighter)
# ============================================================================

class SimpleMultiOCR:
    """
    Simplified multi-OCR system using only Tesseract and EasyOCR
    """
    
    def __init__(self):
        self.reader = None
        self.pytesseract = None
        
    def initialize(self):
        """Initialize OCR engines"""
        try:
            import pytesseract
            self.pytesseract = pytesseract
            print("✓ Tesseract loaded")
        except ImportError:
            print("✗ Tesseract not found")
        
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR loaded")
        except ImportError:
            print("✗ EasyOCR not found")
    
    def extract_text(self, image_path):
        """
        Extract text using both engines and return best result
        """
        if self.pytesseract is None and self.reader is None:
            raise ValueError("No OCR engines initialized")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        results = {}
        
        # Method 1: Tesseract
        if self.pytesseract:
            try:
                # Preprocess for Tesseract
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract with different configurations
                configs = {
                    'default': '--oem 3 --psm 3',
                    'sparse': '--oem 3 --psm 11',
                    'single_block': '--oem 3 --psm 6'
                }
                
                tesseract_results = {}
                for name, config in configs.items():
                    text = self.pytesseract.image_to_string(thresh, config=config)
                    if text.strip():
                        tesseract_results[name] = text
                
                # Use the longest text as best result
                if tesseract_results:
                    best_tess = max(tesseract_results.values(), key=len)
                    results['tesseract'] = best_tess
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        # Method 2: EasyOCR
        if self.reader:
            try:
                easy_result = self.reader.readtext(img, paragraph=True, detail=0)
                if easy_result:
                    results['easyocr'] = '\n'.join(easy_result)
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        # Choose best result
        if not results:
            return None
        
        # Simple heuristic: choose the one with more lines and words
        best_key = max(results.keys(), key=lambda k: len(results[k].split()))
        best_text = results[best_key]
        
        return {
            'text': best_text,
            'source': best_key,
            'all_results': results
        }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example of how to use the multi-model OCR"""
    
    print("=" * 60)
    print("MULTI-MODEL INVOICE OCR SYSTEM")
    print("=" * 60)
    
    # Initialize the advanced system
    print("\n1. Initializing Advanced Multi-Model System...")
    ocr_system = MultiModelInvoiceOCR()
    ocr_system.initialize_models(use_tesseract=True, use_easyocr=True, use_paddleocr=False)
    
    # Process an invoice
    invoice_path = "invoice_02.jpeg"  # Replace with your actual file
    print(f"\n2. Processing invoice: {invoice_path}")
    
    try:
        # Extract data
        result = ocr_system.extract_invoice_data(invoice_path)
        
        print("\n3. EXTRACTION RESULTS:")
        print("-" * 40)
        
        if result:
            for key, value in result.items():
                if key not in ['all_amounts', 'table_lines', 'address_blocks']:     #, 'address_blocks'
                    print(f"{key.replace('_', ' ').title()}: {value}")
            
            print(f"\nAll amounts found: {result.get('all_amounts', [])}")
            # print(f"\nAddress blocks: {result.get('address_blocks', [])}")
            print(f"\nTable lines (first 5): {result.get('table_lines', [])[:5]}")
        else:
            print("No data extracted")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying simplified system...")
        
        # Try simplified system
        simple_ocr = SimpleMultiOCR()
        simple_ocr.initialize()
        
        result = simple_ocr.extract_text(invoice_path)
        if result:
            print(f"\nText extracted using {result['source']}:")
            print("-" * 40)
            print(result['text'][:1500] + "..." if len(result['text']) > 1500 else result['text'])

def batch_process_invoices(invoice_paths):
    """Process multiple invoices"""
    print(f"\nProcessing {len(invoice_paths)} invoices...")
    
    # Initialize simplified system (faster)
    ocr = SimpleMultiOCR()
    ocr.initialize()
    
    results = {}
    
    for path in invoice_paths:
        print(f"\nProcessing: {path}")
        try:
            result = ocr.extract_text(path)
            if result:
                results[path] = {
                    'text': result['text'],
                    'source': result['source']
                }
                print(f"  ✓ Extracted {len(result['text'])} characters using {result['source']}")
            else:
                results[path] = {'error': 'No text extracted'}
                print(f"  ✗ No text extracted")
        except Exception as e:
            results[path] = {'error': str(e)}
            print(f"  ✗ Error: {e}")
    
    return results

# Quick test with a single invoice
if __name__ == "__main__":
    # Example with a single invoice
    # example_usage()
    
    # Or create a quick test script
    print("Quick Invoice OCR Test")
    print("=" * 50)
    
    # Create a minimal working example
    try:
        import cv2
        import pytesseract
        
        # Check if we can read an image
        test_image = "/Users/snehamagdum/Documents/Data/Invoice_03.jpeg"  # Replace with your actual file
        
        # Simple extraction
        img = cv2.imread(test_image)
        if img is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh)
            
            print("\nExtracted Text :")
            print("-" * 40)
            print(text[:1500] + "..." if len(text) > 1500 else text)
            
            # Simple field extraction
            import re
            
            # Look for invoice number
            inv_patterns = [
                r'Invoice No.\s*(?:No\.?|Number|#)?\s*[:]?\s*([A-Z0-9\-]+)',
                r'INV[-_]?(\d+)'
            ]
            
            for pattern in inv_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    print(f"\nInvoice Number: {match.group(1)}")
                    break

            # DATE PATTERNS
            date_patterns = [
                r'(?:Date|Invoice\s*Date|Dated|Issued|Date\s*of\s*Issue)\s*[:]?\s*([\d\/\-\.\sA-Za-z]+)',
                r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
            ]
            
            for pattern in date_patterns:
                dates = re.search(pattern, text, re.IGNORECASE)
                if dates:
                    print(f"\nInvoice Date: {dates.group(1)}")
                    break



                
            # Look for total amount
            total_patterns = [
                r'Total \s*[:]?\s*\$?\s*([\d,]+\.\d{2})',
                r'TOTAL\s*[:]?\s*\$?\s*([\d,]+\.\d{2})'
            ]
            
            for pattern in total_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    print(f"Total Amount: ${match.group(1)}")
                    break
            
            # Find all amounts
            amounts = re.findall(r'[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', text)
            print(f"\nAll amounts found: {amounts}")
            
        else:
            print(f"Cannot read image: {test_image}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nInstallation instructions:")
        print("1. Install Tesseract:")
        print("   - Ubuntu: sudo apt-get install tesseract-ocr")
        print("   - Mac: brew install tesseract")
        print("   - Windows: Download from GitHub")
        print("\n2. Install Python packages:")
        print("   pip install pytesseract opencv-python pillow")
      