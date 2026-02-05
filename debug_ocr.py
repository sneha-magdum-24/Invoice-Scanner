#!/usr/bin/env python3
"""
Debug script to test OCR extraction step by step
"""

import sys
import os
from enhanced_invoice_ocr import EnhancedInvoiceOCR

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    print(f"🔍 Debug OCR for: {image_path}")
    print("=" * 50)
    
    # Initialize with debug enabled
    ocr = EnhancedInvoiceOCR(debug=True)
    
    try:
        # Step 1: Test text extraction
        print("\n📝 Step 1: Text Extraction")
        print("-" * 30)
        
        extracted_text = ocr.extract_text_ensemble(image_path)
        
        if extracted_text:
            print(f"✅ Success: {len(extracted_text)} characters extracted")
            print("\n📄 Extracted Text:")
            print("-" * 40)
            print(extracted_text)
            print("-" * 40)
            
            # Save for inspection
            output_file = f"{os.path.splitext(image_path)[0]}_debug_text.txt"
            with open(output_file, 'w') as f:
                f.write(extracted_text)
            print(f"\n💾 Text saved to: {output_file}")
            
            # Step 2: Test text cleaning
            print("\n🧹 Step 2: Text Cleaning")
            print("-" * 30)
            
            cleaned_text = ocr.clean_and_enhance_text(extracted_text)
            print(f"Cleaned text: {len(cleaned_text)} characters")
            
            if cleaned_text != extracted_text:
                print("\n📄 Cleaned Text:")
                print("-" * 40)
                print(cleaned_text)
                print("-" * 40)
            
            # Step 3: Test Ollama connection
            print("\n🦙 Step 3: Testing Ollama Connection")
            print("-" * 30)
            
            import requests
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    print(f"✅ Ollama connected - {len(models)} models available")
                    for model in models:
                        print(f"   - {model['name']}")
                else:
                    print(f"❌ Ollama API error - Status {response.status_code}")
            except Exception as e:
                print(f"❌ Ollama connection failed: {e}")
                print("💡 Make sure Ollama is running: ollama serve")
                print("💡 Install a model: ollama pull llama3.2:latest")
            
        else:
            print("❌ No text extracted")
            print("💡 Possible issues:")
            print("   - Image quality too poor")
            print("   - OCR engines not installed properly")
            print("   - Image format not supported")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()