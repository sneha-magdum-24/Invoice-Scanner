#!/usr/bin/env python3

import re

class TableExtractor:
    def detect_column_positions(self, header_line):
        """Detect column positions from header line"""
        positions = {}
        header_lower = header_line.lower()
        
        # Find positions of key headers
        desc_match = re.search(r'(description|item|desc)', header_lower)
        qty_match = re.search(r'(qty|quantity|units)', header_lower)
        price_match = re.search(r'(price|rate|unit)', header_lower)
        amount_match = re.search(r'(amount|total)', header_lower)
        
        if desc_match:
            positions['description'] = desc_match.start()
        if qty_match:
            positions['quantity'] = qty_match.start()
        if price_match:
            positions['unit_price'] = price_match.start()
        if amount_match:
            positions['amount'] = amount_match.start()
        
        return positions
    
    def parse_table_row(self, line, column_positions):
        """Parse a table row using column positions"""
        # Extract all numbers from the line
        numbers = re.findall(r'\d+(?:\.\d{1,2})?', line)
        if not numbers:
            return None
        
        # Extract description (text part before numbers)
        desc_match = re.match(r'^([A-Za-z][^0-9$]*)', line)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Clean description
        description = re.sub(r'[^\w\s\-\(\)]', ' ', description).strip()
        description = ' '.join(description.split())
        
        if len(description) < 3:
            return None
        
        # Smart number assignment based on context
        result = {
            'description': description,
            'quantity': '1',
            'unit_price': '0',
            'amount': '0'
        }
        
        # Convert numbers to floats for validation
        float_numbers = []
        for num in numbers:
            try:
                float_numbers.append(float(num))
            except:
                continue
        
        if len(float_numbers) >= 3:
            # Assume: quantity, unit_price, amount
            # Validate: quantity should be small, amount should be qty * unit_price
            qty, price, amount = float_numbers[0], float_numbers[1], float_numbers[2]
            
            # Check if math makes sense
            if abs(qty * price - amount) < 0.01:  # Allow small rounding errors
                result['quantity'] = str(int(qty) if qty.is_integer() else qty)
                result['unit_price'] = str(price)
                result['amount'] = str(amount)
            else:
                # Try different combinations
                if len(float_numbers) >= 2:
                    result['unit_price'] = str(float_numbers[-2])
                    result['amount'] = str(float_numbers[-1])
        
        elif len(float_numbers) == 2:
            # Likely unit_price and amount
            result['unit_price'] = str(float_numbers[0])
            result['amount'] = str(float_numbers[1])
        
        elif len(float_numbers) == 1:
            # Single amount
            result['amount'] = str(float_numbers[0])
            result['unit_price'] = str(float_numbers[0])
        
        return result
    
    def extract_table_data(self, text):
        """Enhanced table extraction with proper column detection"""
        lines = text.split('\n')
        table_data = []
        
        # Find table headers
        header_line_idx = -1
        column_positions = {}
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Look for table headers
            if ('description' in line_lower or 'item' in line_lower) and \
               ('price' in line_lower or 'amount' in line_lower):
                header_line_idx = i
                column_positions = self.detect_column_positions(line)
                break
        
        if header_line_idx == -1:
            print("No table headers found, using fallback method")
            return self.extract_table_fallback(lines)
        
        print(f"Found headers at line {header_line_idx}: {column_positions}")
        
        # Parse data rows after header
        for i in range(header_line_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            # Stop at summary sections
            if any(word in line.lower() for word in ['subtotal', 'tax', 'total', 'discount']):
                break
                
            # Parse line if it contains numbers
            if re.search(r'\d+', line):
                item = self.parse_table_row(line, column_positions)
                if item:
                    table_data.append(item)
                    print(f"Extracted: {item}")
        
        print(f"\nTotal items found: {len(table_data)}")
        return table_data
    
    def extract_table_fallback(self, lines):
        """Fallback table extraction when no headers found"""
        table_data = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip summary lines
            if any(word in line.lower() for word in ['subtotal', 'tax', 'total', 'discount']):
                continue
            
            # Look for lines with descriptions and amounts
            if re.search(r'[a-zA-Z]', line) and re.search(r'\$?\d+(?:\.\d{2})?', line):
                item = self.parse_table_row(line, {})
                if item and item['description']:
                    table_data.append(item)
        
        return table_data

# Test the extraction
if __name__ == "__main__":
    extractor = TableExtractor()
    
    # Test case 1: Standard table format
    print("=== Test Case 1: Standard Table ===")
    sample_text1 = """
Description    Qty    Price    Amount
Service A      1      100.00   100.00
Service B      2      50.00    100.00
Repair work    3      25.00    75.00
"""
    result1 = extractor.extract_table_data(sample_text1)
    for item in result1:
        print(f"  {item}")
    
    # Test case 2: Invoice-like format
    print("\n=== Test Case 2: Invoice Format ===")
    sample_text2 = """
INVOICE #12345
Date: 01/15/2024

Item Description                 Quantity  Unit Price  Amount
Front and rear brake cables      2         30.00       60.00
New set of pedal arms           1         15.00       15.00
Labor 3hrs                      3         45.00       135.00

Subtotal: $210.00
Tax: $15.75
Total: $225.75
"""
    result2 = extractor.extract_table_data(sample_text2)
    for item in result2:
        print(f"  {item}")
    
    # Test case 3: No headers (fallback)
    print("\n=== Test Case 3: No Headers (Fallback) ===")
    sample_text3 = """
Tree removal service $500.00
Cleanup and disposal $150.00
Equipment rental $75.00
"""
    result3 = extractor.extract_table_data(sample_text3)
    for item in result3:
        print(f"  {item}")