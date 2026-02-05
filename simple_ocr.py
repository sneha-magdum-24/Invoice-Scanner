import cv2
import re
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import os
import requests
import json
import random
from paddleocr import PaddleOCR

class MobileInvoiceOCR:
    """
    Mobile-optimized OCR system for invoice processing:
    OpenCV Cleanup → Document Crop → Deskew → PaddleOCR → LLM Parsing → JSON
    """
   
    def __init__(self, debug=False):
        self.debug = debug
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    def opencv_cleanup(self, image_path):
        """Step 1: OpenCV cleanup for mobile images"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize if too large
        h, w = img.shape[:2]
        if w > 1500:
            scale = 1500 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_document(self, image):
        """Step 2: Document detection and cropping"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest rectangular contour
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4 and area > image.shape[0] * image.shape[1] * 0.1:
                    largest_contour = approx
                    max_area = area
        
        if largest_contour is not None:
            # Order points for perspective transform
            pts = largest_contour.reshape(4, 2)
            rect = self.order_points(pts)
            
            # Apply perspective transform
            warped = self.four_point_transform(image, rect)
            return warped
        
        return image
    
    def order_points(self, pts):
        """Order points for perspective transform"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def four_point_transform(self, image, rect):
        """Apply perspective transform"""
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def deskew_image(self, image):
        """Step 3: Deskew the document"""
        # Detect lines using HoughLines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for line in lines[:10]:  # Use first 10 lines
                rho, theta = line[0]  # Extract from nested array
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Only rotate if significant skew
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        
        return image
    
    def extract_text_paddleocr(self, image):
        """Step 4: Extract text using PaddleOCR"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run PaddleOCR
        result = self.ocr.predict(image_rgb)
        
        # Extract text with layout preservation
        text_blocks = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, confidence) = line
                if confidence > 0.5:  # Filter low confidence
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    text_blocks.append((y_center, text))
        
        # Sort by y-coordinate and join
        text_blocks.sort(key=lambda x: x[0])
        extracted_text = '\n'.join([text for _, text in text_blocks])
        
        return extracted_text
    
    def extract_text(self, file_path):
        """Main text extraction pipeline for mobile images"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        else:
            # Mobile-optimized image processing pipeline
            print("Step 1: OpenCV cleanup...")
            cleaned_img = self.opencv_cleanup(file_path)
            
            print("Step 2: Document detection and cropping...")
            cropped_img = self.detect_document(cleaned_img)
            
            print("Step 3: Deskewing...")
            deskewed_img = self.deskew_image(cropped_img)
            
            if self.debug:
                cv2.imwrite('debug_processed.jpg', deskewed_img)
                print("Debug: Saved processed image as debug_processed.jpg")
            
            print("Step 4: PaddleOCR text extraction...")
            text = self.extract_text_paddleocr(deskewed_img)
            
            return text
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    

    

    

    
    def extract_invoice_data_mobile(self, file_path):
        """Mobile-optimized invoice processing pipeline"""
        print(f"Processing mobile image: {file_path}")
        
        # Step 1-4: Extract text using mobile pipeline
        extracted_text = self.extract_text(file_path)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            print("Text extraction failed")
            return None
        
        print(f"Extracted {len(extracted_text)} characters")
        
        # Step 5: LLM processing
        print("Step 5: LLM parsing...")
        result = self.process_with_ollama(extracted_text)
        
        if result:
            print("Step 6: JSON output complete")
            return result
        else:
            print("LLM processing failed")
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
    


# Usage example
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Invoice OCR Extraction')
    parser.add_argument('file_path', nargs='?', help='Path to invoice image or PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Initialize OCR system
    ocr = MobileInvoiceOCR(debug=args.debug)
    
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
        # Use the mobile-optimized approach
        result = ocr.extract_invoice_data_mobile(invoice_path)
        
        if result:
            print("\nMobile OCR Extraction Result:")
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