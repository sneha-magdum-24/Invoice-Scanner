import cv2
import numpy as np
import re
import os
import requests
import json
import base64
from PIL import Image

class DeepSeekInvoiceOCR:
    def __init__(self, api_key=None):
        self.deepseek_api_key = api_key or "YOUR_DEEPSEEK_API_KEY"  # Replace with your actual API key
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text_with_deepseek(self, image_path):
        """Extract text using DeepSeek Vision API"""
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-vl-7b-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this invoice image. Preserve the layout and structure. Include all numbers, descriptions, and table data exactly as they appear."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(self.deepseek_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result['choices'][0]['message']['content']
                return extracted_text
            else:
                print(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"DeepSeek OCR error: {e}")
            return None
    
    def process_with_ollama(self, text):
        """Process text with Ollama"""
        prompt = f"""Extract invoice data as JSON. Find:
- vendor_name: Company name at top
- date: Invoice date
- items: List with item_name, quantity, unit_price, amount
- subtotal: Subtotal amount
- tax_amount: Tax amount  
- total: Total amount

Look for table columns like QTY, DESCRIPTION, UNIT PRICE, AMOUNT.
Extract quantities, unit prices, and line amounts.

Text:
{text}

Return only JSON:"""

        try:
            response = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                }, timeout=30)
            
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
            print(f"Ollama error: {e}")
        
        return None
    
    def extract_invoice_data(self, image_path):
        """Main extraction function"""
        print(f"Processing: {image_path}")
        
        # Extract text using DeepSeek
        text = self.extract_text_with_deepseek(image_path)
        
        if not text:
            print("DeepSeek OCR failed")
            return None
            
        print(f"Extracted {len(text)} characters")
        
        # Save extracted text for debugging
        with open(f"{os.path.splitext(image_path)[0]}_deepseek_text.txt", 'w') as f:
            f.write(text)
        print("Raw text saved")
        
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
        print("Usage: python deepseek_ocr.py <image_path> [api_key]")
        return
    
    image_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    ocr = DeepSeekInvoiceOCR(api_key)
    result = ocr.extract_invoice_data(image_path)
    
    if result:
        print("\nExtracted Data:")
        print(json.dumps(result, indent=2))
        
        # Save result
        output_file = f"{os.path.splitext(image_path)[0]}_deepseek_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        print("Extraction failed")

if __name__ == "__main__":
    main()