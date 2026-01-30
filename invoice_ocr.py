import cv2
import re
import pytesseract
from PIL import Image
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import os



import requests
import json

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
        """Basic image preprocessing for better OCR"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold for better text recognition
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
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
        """Extract text from image using Tesseract OCR"""
        processed_img = self.preprocess_image(image_path)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed_img)
        
        # Extract text with optimal configuration for invoices
        text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')
        
        return text
    
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
    
    def process_with_ollama(self, invoice_text, table_rows=""):
        """Process invoice text using Ollama with Gemma2 2B model"""
        
        # Get the extraction prompt
        prompt = self.extract_invoice_data_with_prompt(invoice_text, table_rows)
        
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "gemma2:2b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent output
                "top_p": 0.9
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
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = raw_response[start_idx:end_idx]
                    parsed_json = json.loads(json_str)
                    return parsed_json
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
        
        prompt = f"""You are an invoice extraction engine.

Return ONLY valid JSON. Do not include markdown. Do not include any explanation.
Do not add extra keys. Output must match this exact schema:

{{
  "vendor_name": null,
  "date": null,
  "items": [
    {{ "item_name": null, "unit_price": null, "quantity": null, "amount": null }}
  ],
  "subtotal": null,
  "tax_percent": null,
  "discount_percent": null,
  "total": null
}}

Rules:
1) Use null if a value is not present or not confident.
2) All money numbers must be plain numbers only (no currency symbol, no commas). Example: 5659.5
3) date must be ISO format YYYY-MM-DD if possible, else null.
4) items:
   - item_name must be text.
   - unit_price, quantity, amount must be numbers or null.
   - If amount is missing but unit_price and quantity exist: amount = unit_price * quantity
   - If unit_price is missing but amount and quantity exist and quantity > 0: unit_price = amount / quantity
5) subtotal:
   - Prefer explicit "Subtotal" value if present.
   - If not present, subtotal = sum(items.amount) if items.amount exists.
6) tax_percent:
   - Use the tax percent if shown (e.g., "GST 18%").
   - If only tax amount is present and subtotal > 0, tax_percent = (tax_amount / subtotal) * 100
   - If neither, null.
7) discount_percent:
   - If discount is shown as percentage (e.g., "10% off"), use that percentage.
   - If discount is a flat amount (e.g., "Discount $500"), store the flat amount as negative number.
   - If neither, null.
8) total:
   - Prefer explicit "Total / Amount Due / Grand Total".
   - If missing and subtotal exists, total = subtotal + tax_amount - discount_amount (only if those amounts exist).
   - Otherwise null.
9) Vendor name:
   - Extract the main business/company name that is providing the service (not the customer).
   - Look for company names near the top, often in larger text or headers.
   - Avoid customer names, addresses, or recipient information.
   - Common patterns: "[Company Name] LLC", "[Company Name] Inc", or standalone business names.
10) Item descriptions:
   - Use the full descriptive text for each service/product, not just generic categories.
   - Include specific details when available (e.g., "Cherry Tree Removal (North Of The Garage)" not just "Tree Removal").
   - Preserve important descriptive information that differentiates items.
INVOICE TEXT (may include OCR text):
<<<
{invoice_text}
>>>

OPTIONAL TABLE ROWS (if you have extracted rows separately):
<<<
{table_rows}
>>>"""
        
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
    # Initialize OCR system
    ocr = FocusedInvoiceOCR()
    
    # Process invoice
    invoice_path = "/Users/snehamagdum/Documents/Data/IMG_0182.jpg"  # Update with your path
    
    try:
        # Extract text from invoice
        text = ocr.extract_text(invoice_path)
        
        print("Extracted Text:")
        print("=" * 50)
        print(text[:1500] + "..." if len(text) > 1500 else text)
        print("=" * 50)
        
        # Process with Ollama
        print("\nProcessing with Ollama (Gemma2 2B)...")
        ollama_result = ocr.process_with_ollama(text)
        
        if ollama_result:
            print("\nOllama Extraction Result:")
            print("=" * 50)
            print(json.dumps(ollama_result, indent=2))
            print("=" * 50)
            
            # Save Ollama result
            with open('ollama_invoice_data.json', 'w') as f:
                json.dump(ollama_result, f, indent=2)
            print("\nOllama result saved to 'ollama_invoice_data.json'")
            return ollama_result
        else:
            print("Failed to process with Ollama")
            return None
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()