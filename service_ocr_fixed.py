import cv2
import re
import os
import requests
import json
import random

# Optional imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

class ServiceInvoiceOCR:
    def __init__(self):
        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            self.reader = None
    
    def extract_text(self, file_path):
        """Extract text from image or PDF"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf' and FITZ_AVAILABLE:
            return self.extract_text_from_pdf(file_path)
        else:
            return self.extract_text_from_image(file_path)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text using EasyOCR"""
        if not self.reader:
            print("EasyOCR not available")
            return ""
        
        try:
            results = self.reader.readtext(image_path)
            high_conf_results = [r for r in results if r[2] > 0.3]
            extracted_text = self.reconstruct_lines(high_conf_results)
            print(f"EasyOCR extracted {len(extracted_text)} characters")
            return extracted_text
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def reconstruct_lines(self, ocr_results):
        """Reconstruct text lines from OCR results"""
        if not ocr_results:
            return ""
        
        # Sort by Y coordinate
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
        
        lines = []
        current_line = []
        current_y = None
        
        for bbox, text, conf in sorted_results:
            y_top = bbox[0][1]
            
            if current_y is None or abs(y_top - current_y) < 20:
                current_line.append((bbox[0][0], text))
                current_y = y_top if current_y is None else current_y
            else:
                if current_line:
                    current_line.sort(key=lambda x: x[0])
                    line_text = ' '.join([t for _, t in current_line])
                    lines.append(line_text)
                current_line = [(bbox[0][0], text)]
                current_y = y_top
        
        if current_line:
            current_line.sort(key=lambda x: x[0])
            line_text = ' '.join([t for _, t in current_line])
            lines.append(line_text)
        
        return '\n'.join(lines)
    
    def process_with_ollama(self, text):
        """Process with Ollama LLM"""
        prompt = f"""Extract invoice data as JSON. Focus on SERVICE ITEMS with pricing.

RULES:
1. Extract service description, quantity, unit_price, amount for each item
2. Include ALL services with pricing
3. Return ONLY valid JSON, no explanations

JSON format:
{{
  "vendor_name": "Company Name",
  "date": "Date", 
  "items": [
    {{
      "item_name": "Service Description",
      "unit_price": 100.00,
      "quantity": 1,
      "amount": 100.00
    }}
  ],
  "subtotal": 0.00,
  "tax_amount": 0.00,
  "total": 0.00
}}

Text:
{text}

JSON:"""

        try:
            response = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1000}
                }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                print(f"Raw Ollama response: {raw_response[:200]}...")
                
                # Clean response
                raw_response = raw_response.replace('&quot;', '"').replace('&gt;', '>').replace('&lt;', '<')
                
                # Extract JSON
                start = raw_response.find('{')
                if start == -1:
                    print("No JSON found in response")
                    return None
                
                # Find matching closing brace
                brace_count = 0
                end = start
                for i, char in enumerate(raw_response[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                json_str = raw_response[start:end]
                print(f"Extracted JSON: {json_str[:100]}...")
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Problematic JSON: {json_str}")
                    return None
            else:
                print(f"Ollama API error: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def extract_invoice_data(self, image_path):
        """Main extraction function"""
        print(f"Processing: {image_path}")
        
        # Extract text
        text = self.extract_text(image_path)
        if not text or len(text) < 20:
            print("Insufficient text extracted")
            return None
        
        print(f"Extracted {len(text)} characters")
        
        # Save raw text
        with open(f"{os.path.splitext(image_path)[0]}_text.txt", 'w') as f:
            f.write(text)
        
        # Process with LLM
        result = self.process_with_ollama(text)
        
        if result:
            print("Extraction successful")
            return result
        else:
            print("LLM processing failed")
            return None

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python service_ocr_fixed.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    ocr = ServiceInvoiceOCR()
    result = ocr.extract_invoice_data(image_path)
    
    if result:
        print("\nExtracted Services:")
        print("=" * 80)
        
        items = result.get('items', [])
        if items:
            print(f"{'Description':<40} {'Qty':<5} {'Unit Price':<12} {'Amount':<10}")
            print("-" * 80)
            for item in items:
                desc = item['item_name'][:37] + "..." if len(item['item_name']) > 40 else item['item_name']
                qty = item.get('quantity', 1)
                price = item.get('unit_price', 0)
                amount = item.get('amount', 0)
                print(f"{desc:<40} {qty:<5} ${price:<11.2f} ${amount:<10.2f}")
        
        print("\n" + "=" * 50)
        print(f"Vendor: {result.get('vendor_name', 'N/A')}")
        print(f"Date: {result.get('date', 'N/A')}")
        print(f"Subtotal: ${result.get('subtotal', 0)}")
        print(f"Tax: ${result.get('tax_amount', 0)}")
        print(f"Total: ${result.get('total', 0)}")
        
        # Save result
        output_file = f"{os.path.splitext(image_path)[0]}_services.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        print("Extraction failed")

if __name__ == "__main__":
    main()