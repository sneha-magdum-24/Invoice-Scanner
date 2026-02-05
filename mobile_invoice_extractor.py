import cv2
import easyocr
import requests
import json
import re
import os
import numpy as np

class MobileInvoiceExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def extract_text_from_image(self, image_path):
        """Extract text from mobile camera photo"""
        try:
            results = self.reader.readtext(image_path)
            high_conf_results = [r for r in results if r[2] > 0.3]
            return self.reconstruct_lines(high_conf_results)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def reconstruct_lines(self, ocr_results):
        """Reconstruct text lines from OCR results"""
        if not ocr_results:
            return ""
        
        # Sort by Y coordinate (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
        
        lines = []
        current_line = []
        
        for item in sorted_results:
            bbox, text, conf = item
            y_top = bbox[0][1]
            
            if not current_line:
                current_line.append(item)
                continue
            
            # Check if item is on same line (similar Y coordinate)
            avg_y = sum(x[0][0][1] for x in current_line) / len(current_line)
            
            if abs(y_top - avg_y) < 20:  # Same line
                current_line.append(item)
            else:
                # New line
                lines.append(current_line)
                current_line = [item]
        
        if current_line:
            lines.append(current_line)
        
        # Join text for each line
        final_text = ""
        for line in lines:
            line.sort(key=lambda x: x[0][0][0])  # Sort by X coordinate
            line_text = "   ".join([x[1] for x in line])
            final_text += line_text + "\n"
        
        return final_text
    
    def extract_invoice_data(self, image_path):
        """Extract structured invoice data"""
        print(f"Processing: {image_path}")
        
        # Extract text
        text = self.extract_text_from_image(image_path)
        if not text:
            return None
        
        print("Extracted text:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        
        # Process with LLM
        return self.process_with_llm(text)
    
    def process_with_llm(self, invoice_text):
        """Process text with Ollama LLM"""
        prompt = f"""Extract invoice data from this text. Return valid JSON only.

{{
  "vendor_name": null,
  "date": null,
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "total": null,
  "items": [
    {{"item_name": null, "quantity": null, "unit_price": null, "amount": null}}
  ]
}}

Rules:
1. Find vendor/company name at top
2. Find date (issued/due date)
3. For each item row, read the ACTUAL service description as item_name
4. Read quantity from quantity column (numbers like 1, 2, 3, etc.)
5. Read unit_price from unit price column
6. Read amount from amount column
7. Do NOT use time durations ("6 hour", "3 hour") as item names
8. Look for actual service descriptions like "Web elements design", "UX design"
9. Remove $ signs from numbers

Example:
If you see: "Web elements design | 2 | $250.00 | $500.00"
Extract: item_name="Web elements design", quantity=2, unit_price=250.00, amount=500.00

Text:
{invoice_text}"""

        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            })
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                # Extract JSON
                start = raw_response.find('{')
                end = raw_response.rfind('}') + 1
                
                if start != -1 and end > start:
                    json_str = raw_response[start:end]
                    return json.loads(json_str)
            
        except Exception as e:
            print(f"LLM processing error: {e}")
        
        return None

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mobile_invoice_extractor.py <image_path>")
        return
    
    extractor = MobileInvoiceExtractor()
    result = extractor.extract_invoice_data(sys.argv[1])
    
    if result:
        print("\nExtracted Invoice Data:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        
        # Save result
        output_file = f"{os.path.splitext(sys.argv[1])[0]}_extracted.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        print("Failed to extract invoice data")

if __name__ == "__main__":
    main()