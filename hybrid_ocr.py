import cv2
import numpy as np
import re
import os
import requests
import json
import base64
from PIL import Image

class HybridInvoiceOCR:
    def __init__(self, deepseek_api_key=None):
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_with_deepseek(self, image_path):
        """Try DeepSeek first"""
        if not self.deepseek_api_key or self.deepseek_api_key == "YOUR_DEEPSEEK_API_KEY":
            return None
            
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-vl-7b-chat",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this invoice image. Preserve layout and structure."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(self.deepseek_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"DeepSeek failed: {e}")
        
        return None
    
    def extract_with_easyocr(self, image_path):
        """Fallback to EasyOCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            
            # Preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR
            results = reader.readtext(enhanced)
            
            # Reconstruct text with layout
            text_blocks = []
            for bbox, text, conf in results:
                if conf > 0.3:
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    x_left = bbox[0][0]
                    text_blocks.append((y_center, x_left, text))
            
            # Sort by position
            text_blocks.sort(key=lambda x: (x[0], x[1]))
            
            # Group into lines
            lines = []
            current_line = []
            current_y = None
            
            for y, x, text in text_blocks:
                if current_y is None or abs(y - current_y) < 20:
                    current_line.append((x, text))
                    current_y = y if current_y is None else current_y
                else:
                    if current_line:
                        current_line.sort(key=lambda x: x[0])
                        line_text = ' '.join([t for _, t in current_line])
                        lines.append(line_text)
                    current_line = [(x, text)]
                    current_y = y
            
            if current_line:
                current_line.sort(key=lambda x: x[0])
                line_text = ' '.join([t for _, t in current_line])
                lines.append(line_text)
            
            return '\n'.join(lines)
            
        except ImportError:
            print("EasyOCR not available")
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        return None
    
    def extract_with_tesseract(self, image_path):
        """Fallback to Tesseract"""
        try:
            import pytesseract
            
            # Preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with different configs
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 3',
                '--oem 3 --psm 11'
            ]
            
            best_text = ""
            for config in configs:
                text = pytesseract.image_to_string(thresh, config=config)
                if len(text) > len(best_text):
                    best_text = text
            
            return best_text
            
        except ImportError:
            print("Tesseract not available")
        except Exception as e:
            print(f"Tesseract failed: {e}")
        
        return None
    
    def extract_text(self, image_path):
        """Try multiple OCR methods"""
        print("Trying DeepSeek OCR...")
        text = self.extract_with_deepseek(image_path)
        if text:
            print("DeepSeek OCR successful")
            return text
        
        print("Trying EasyOCR...")
        text = self.extract_with_easyocr(image_path)
        if text:
            print("EasyOCR successful")
            return text
        
        print("Trying Tesseract...")
        text = self.extract_with_tesseract(image_path)
        if text:
            print("Tesseract successful")
            return text
        
        print("All OCR methods failed")
        return None
    
    def process_with_ollama(self, text):
        """Process with Ollama"""
        prompt = f"""Extract invoice data as JSON:
- vendor_name: Company name
- date: Invoice date  
- items: [{{item_name, quantity, unit_price, amount}}]
- subtotal, tax_amount, total

Text:
{text}

JSON only:"""

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
        
        # Extract text
        text = self.extract_text(image_path)
        
        if not text:
            print("Text extraction failed")
            return None
            
        print(f"Extracted {len(text)} characters")
        
        # Save text
        with open(f"{os.path.splitext(image_path)[0]}_extracted_text.txt", 'w') as f:
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
        print("Usage: python hybrid_ocr.py <image_path> [deepseek_api_key]")
        return
    
    image_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    ocr = HybridInvoiceOCR(api_key)
    result = ocr.extract_invoice_data(image_path)
    
    if result:
        print("\nExtracted Data:")
        print(json.dumps(result, indent=2))
        
        output_file = f"{os.path.splitext(image_path)[0]}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        print("Extraction failed")

if __name__ == "__main__":
    main()