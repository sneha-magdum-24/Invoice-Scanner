import os
import json
import requests
import random
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe

class MinerUInvoiceOCR:
    """
    MinerU-based OCR system for mobile invoice processing:
    Mobile Photo/PDF → MinerU (OCR + Layout) → Clean Text + Tables → LLM/Rules → Invoice JSON
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def process_with_mineru(self, file_path):
        """Step 1-2: Process with MinerU for OCR + Layout detection"""
        print("Step 1-2: MinerU OCR + Layout processing...")
        
        try:
            # Initialize MinerU pipeline
            pipe = UNIPipe(
                pdf_path=file_path,
                output_dir="./mineru_output",
                image_writer="img",
                table_writer="csv"
            )
            
            # Process the document
            pipe_ret = pipe.pipe_classify()
            
            if pipe_ret:
                pipe_ret = pipe.pipe_analyze()
                pipe_ret = pipe.pipe_parse()
                
                # Extract text and tables
                md_content = pipe.pipe_mk()
                
                return md_content
            else:
                print("MinerU processing failed")
                return None
                
        except Exception as e:
            print(f"MinerU error: {e}")
            # Fallback to OCR-only approach
            return self.fallback_ocr(file_path)
    
    def fallback_ocr(self, file_path):
        """Fallback OCR processing if MinerU fails"""
        print("Using fallback OCR processing...")
        
        try:
            # Use OCR pipe for images
            ocr_pipe = OCRPipe(
                image_path=file_path,
                output_dir="./ocr_output"
            )
            
            result = ocr_pipe.pipe_ocr()
            return result.get('content', '')
            
        except Exception as e:
            print(f"Fallback OCR error: {e}")
            return ""
    
    def clean_text_and_tables(self, md_content):
        """Step 3: Clean extracted text and tables"""
        print("Step 3: Cleaning text and tables...")
        
        if not md_content:
            return ""
        
        # Basic cleaning
        lines = md_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove markdown formatting
            line = line.replace('**', '').replace('*', '')
            line = line.replace('|', ' ')  # Clean table separators
            
            # Fix common OCR errors
            line = line.replace('8', '$', 1) if line.startswith('8') and any(c.isdigit() for c in line[1:]) else line
            line = line.replace('S', '$', 1) if line.startswith('S') and any(c.isdigit() for c in line[1:]) else line
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        if self.debug:
            print("Cleaned text:")
            print("-" * 50)
            print(cleaned_text)
            print("-" * 50)
        
        return cleaned_text
    
    def process_with_llm(self, cleaned_text):
        """Step 4: Process with LLM for structured extraction"""
        print("Step 4: LLM processing...")
        
        prompt = f"""Extract invoice data from the following text. Return ONLY valid JSON without markdown blocks.

Schema:
{{
  "vendor_name": null,
  "date": null,
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "total": null,
  "items": [
    {{"item_name": null, "unit_price": null, "quantity": null, "amount": null}}
  ]
}}

Rules:
1. Extract vendor name from the top of the document
2. Find date in format MM/DD/YYYY or similar
3. Extract all line items with prices
4. Get subtotal, tax, discount, and total amounts
5. Convert all amounts to numbers (e.g., 1500.00 not "1,500.00")
6. Discount is NOT an item - put in discount_amount field

Text:
{cleaned_text}"""

        return self.call_ollama(prompt)
    
    def call_ollama(self, prompt):
        """Call Ollama API for LLM processing"""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1000,
                "seed": random.randint(0, 100000)
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get('response', '')
            
            # Extract JSON
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = raw_response[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix JSON
                    return self.fix_json(json_str)
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama API error: {e}")
            return None
    
    def fix_json(self, json_str):
        """Fix common JSON formatting issues"""
        try:
            # Balance brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            return json.loads(json_str)
        except:
            return None
    
    def extract_invoice_data(self, file_path):
        """Main processing pipeline"""
        print(f"Processing mobile invoice: {file_path}")
        
        # Step 1-2: MinerU processing
        md_content = self.process_with_mineru(file_path)
        
        if not md_content:
            print("MinerU processing failed")
            return None
        
        # Step 3: Clean text and tables
        cleaned_text = self.clean_text_and_tables(md_content)
        
        if not cleaned_text:
            print("Text cleaning failed")
            return None
        
        # Step 4: LLM processing
        result = self.process_with_llm(cleaned_text)
        
        if result:
            print("Step 5: Invoice JSON generated successfully")
            return result
        else:
            print("LLM processing failed")
            return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MinerU Invoice OCR Extraction')
    parser.add_argument('file_path', nargs='?', help='Path to invoice image or PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Initialize OCR system
    ocr = MinerUInvoiceOCR(debug=args.debug)
    
    # Determine file path
    if args.file_path:
        invoice_path = args.file_path
    else:
        # Scan for files
        potential_files = [f for f in os.listdir('.') 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))]
        
        if not potential_files:
            print("No image/PDF files found.")
            print("Usage: python mineru_invoice_ocr.py <path_to_invoice>")
            return
            
        print("Available files:")
        for i, f in enumerate(potential_files):
            print(f"{i+1}. {f}")
        
        try:
            selection = input("\nSelect file (or 'q' to quit): ")
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
        # Process invoice
        result = ocr.extract_invoice_data(invoice_path)
        
        if result:
            print("\nMinerU Invoice Extraction Result:")
            print("=" * 50)
            print(json.dumps(result, indent=2))
            print("=" * 50)
            
            # Save result
            output_name = f"{os.path.splitext(os.path.basename(invoice_path))[0]}_mineru.json"
            with open(output_name, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to '{output_name}'")
            
        else:
            print("Failed to process invoice")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()