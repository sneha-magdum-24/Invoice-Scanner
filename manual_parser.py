import cv2
import easyocr
import json
import re

class InvoiceOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extract_all_text(self, image_path):
        """Extract all text from image"""
        results = self.reader.readtext(image_path)
        
        # Sort by Y coordinate then X coordinate for proper reading order
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        
        # Get all text with confidence > 0.2
        all_text = []
        for bbox, text, conf in results:
            if conf > 0.2:
                all_text.append(text)
        
        return '\n'.join(all_text)
    
    def extract_vendor_name(self, text):
        """Extract vendor/company name"""
        lines = text.split('\n')
        
        # Look for specific company patterns first
        company_patterns = [
            r'([A-Z][A-Za-z\s&.,]+(?:Inc|LLC|Ltd|Corp|Company|Co\.|Corporation|LTD|INC))',
            r'([A-Z][A-Za-z\s&.,]{10,50})',  # Capitalized company names
            r'([A-Z\s]{5,30})',  # All caps company names
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1).strip()
                if len(company) > 5 and 'INVOICE' not in company.upper():
                    return company
        
        # Fallback: first meaningful line that's not INVOICE
        for line in lines[:10]:
            line = line.strip()
            if (len(line) > 5 and not line.isdigit() and 
                'INVOICE' not in line.upper() and 
                'PAID' not in line.upper() and
                not re.match(r'^\d+', line)):
                return line
        return None
    
    def extract_date(self, text):
        """Extract date in various formats"""
        date_patterns = [
            r'(?:Date[:\s]*)?([01]?\d/[0-3]?\d/\d{4})',  # MM/DD/YYYY or M/D/YYYY
            r'(?:Date[:\s]*)?([0-3]?\d/[01]?\d/\d{4})',  # DD/MM/YYYY or D/M/YYYY
            r'(?:Date[:\s]*)?([01]?\d-[0-3]?\d-\d{4})',  # MM-DD-YYYY
            r'(?:Date[:\s]*)?(\d{4}-[01]?\d-[0-3]?\d)',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # Generic MM/DD/YYYY
        ]
        
        # Look for dates near "INVOICE" or at the top
        lines = text.split('\n')
        for i, line in enumerate(lines[:15]):  # Check first 15 lines
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    date_str = match.group(1)
                    # Validate it's a reasonable date
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3 and all(part.isdigit() for part in parts):
                            return date_str
        return None
    
    def extract_items(self, text):
        """Extract table items with proper column parsing"""
        items = []
        lines = text.split('\n')
        
        # Find table section (after headers like QTY, COST, PRICE)
        table_start = -1
        for i, line in enumerate(lines):
            if any(header in line.upper() for header in ['QTY', 'COST', 'PRICE', 'AMOUNT']):
                table_start = i + 1
                break
        
        # If no headers found, start from beginning
        if table_start == -1:
            table_start = 0
        
        # Process lines that contain service descriptions and prices
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines and summary lines
            if not line or any(word in line.upper() for word in ['SUBTOTAL', 'TOTAL', 'TAX', 'DISCOUNT', 'TERMS', 'NOTES']):
                continue
            
            # Look for service lines with descriptions and amounts
            if any(keyword in line.upper() for keyword in ['SERVICE', 'MICE', 'INITIAL', 'REPAIR', 'LABOR']) or '$' in line:
                
                # Extract description (everything before numbers/prices)
                desc_parts = []
                words = line.split()
                
                for word in words:
                    if '$' in word or word.replace('.', '').isdigit():
                        break
                    desc_parts.append(word)
                
                description = ' '.join(desc_parts).strip()
                
                # Extract all numbers from the line
                numbers = re.findall(r'\b(\d+(?:\.\d{1,2})?)\b', line)
                prices = re.findall(r'\$(\d+(?:\.\d{2})?)', line)
                
                if description and (numbers or prices):
                    # Default values
                    quantity = 1
                    unit_price = 0.0
                    amount = 0.0
                    
                    # Parse quantities and prices
                    if prices:
                        if len(prices) >= 2:
                            # Multiple prices: likely unit_price and amount
                            unit_price = float(prices[0])
                            amount = float(prices[-1])
                        else:
                            # Single price: use as both unit_price and amount
                            unit_price = amount = float(prices[0])
                    
                    # Look for quantity in numbers (usually small integers)
                    if numbers:
                        for num in numbers:
                            num_val = float(num)
                            if 1 <= num_val <= 100 and num_val.is_integer():
                                quantity = int(num_val)
                                break
                    
                    # Validate and add item
                    if description and (unit_price > 0 or amount > 0):
                        items.append({
                            "item_name": description,
                            "unit_price": unit_price,
                            "quantity": quantity,
                            "amount": amount if amount > 0 else unit_price
                        })
        
        return items
    
    def extract_subtotal(self, text):
        """Extract subtotal"""
        patterns = [
            r'Subtotal[:\s]*\$(\d+(?:\.\d{2})?)',
            r'Sub[\s-]*Total[:\s]*\$(\d+(?:\.\d{2})?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0
    
    def extract_tax(self, text):
        """Extract tax amount"""
        patterns = [
            r'Tax[:\s]*\$(\d+(?:\.\d{2})?)',
            r'Sales Tax[:\s]*\$(\d+(?:\.\d{2})?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0
    
    def extract_discount(self, text):
        """Extract discount amount"""
        patterns = [
            r'Discount[:\s]*\$(\d+(?:\.\d{2})?)',
            r'Disc[:\s]*\$(\d+(?:\.\d{2})?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0
    
    def extract_total(self, text):
        """Extract total amount"""
        patterns = [
            r'Total[:\s]*\$(\d+(?:\.\d{2})?)',
            r'Amount Due[:\s]*\$(\d+(?:\.\d{2})?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0
    
    def process_invoice(self, image_path):
        """Complete invoice processing"""
        # Extract all text
        text = self.extract_all_text(image_path)
        print(f"Extracted text ({len(text)} chars):")
        print("-" * 50)
        print(text)
        print("-" * 50)
        
        # Parse invoice data
        result = {
            "vendor_name": self.extract_vendor_name(text),
            "date": self.extract_date(text),
            "items": self.extract_items(text),
            "subtotal": self.extract_subtotal(text),
            "tax_amount": self.extract_tax(text),
            "discount_amount": self.extract_discount(text),
            "total": self.extract_total(text)
        }
        
        return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python manual_parser.py <image_path>")
        sys.exit(1)
    
    ocr = InvoiceOCR()
    result = ocr.process_invoice(sys.argv[1])
    
    print("\nExtracted Invoice Data:")
    print(json.dumps(result, indent=2))