#!/usr/bin/env python3
"""
Example usage of the Enhanced Invoice OCR system.
Demonstrates various ways to use the system programmatically.
"""

import os
import json
from enhanced_invoice_ocr import EnhancedInvoiceOCR

def example_basic_usage():
    """Basic usage example"""
    print("🔹 Basic Usage Example")
    print("-" * 30)
    
    # Initialize OCR system
    ocr = EnhancedInvoiceOCR(debug=True)
    
    # Process an invoice
    invoice_path = "sample_invoice.png"
    
    if not os.path.exists(invoice_path):
        print(f"❌ Sample invoice not found: {invoice_path}")
        print("💡 Run test_installation.py first to create a sample invoice")
        return
    
    # Extract data
    result = ocr.extract_invoice_data(invoice_path, model_type="ollama")
    
    if result:
        print("✅ Extraction successful!")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Extraction failed")

def example_batch_processing():
    """Batch processing example"""
    print("\n🔹 Batch Processing Example")
    print("-" * 30)
    
    # Initialize OCR system (no debug for batch)
    ocr = EnhancedInvoiceOCR(debug=False)
    
    # Find all invoice files in current directory
    invoice_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp')
    invoice_files = [f for f in os.listdir('.') if f.lower().endswith(invoice_extensions)]
    
    if not invoice_files:
        print("❌ No invoice files found in current directory")
        return
    
    print(f"📁 Found {len(invoice_files)} invoice files")
    
    results = []
    
    for i, filename in enumerate(invoice_files, 1):
        print(f"\n📄 Processing {i}/{len(invoice_files)}: {filename}")
        
        try:
            result = ocr.extract_invoice_data(filename, model_type="ollama")
            
            if result:
                # Add filename to result
                result['source_file'] = filename
                results.append(result)
                
                print(f"   ✅ Success - Vendor: {result.get('vendor_name', 'Unknown')}")
                print(f"   💰 Total: ${result.get('total', 0)}")
            else:
                print(f"   ❌ Failed to extract data")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save batch results
    if results:
        output_file = "batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Batch processing complete!")
        print(f"   Processed: {len(results)}/{len(invoice_files)} files")
        print(f"   Results saved to: {output_file}")
    else:
        print("\n❌ No successful extractions")

def example_text_only_extraction():
    """Text extraction only (no LLM processing)"""
    print("\n🔹 Text-Only Extraction Example")
    print("-" * 30)
    
    ocr = EnhancedInvoiceOCR(debug=True)
    
    invoice_path = "sample_invoice.png"
    
    if not os.path.exists(invoice_path):
        print(f"❌ Sample invoice not found: {invoice_path}")
        return
    
    # Extract just the text (no LLM processing)
    extracted_text = ocr.extract_text_ensemble(invoice_path)
    
    if extracted_text:
        print("✅ Text extraction successful!")
        print("\n📝 Extracted Text:")
        print("-" * 40)
        print(extracted_text)
        print("-" * 40)
        
        # Save raw text
        with open("extracted_text.txt", 'w') as f:
            f.write(extracted_text)
        print("\n💾 Raw text saved to: extracted_text.txt")
    else:
        print("❌ Text extraction failed")

def example_different_models():
    """Example using different LLM models"""
    print("\n🔹 Different LLM Models Example")
    print("-" * 30)
    
    ocr = EnhancedInvoiceOCR(debug=False)
    
    invoice_path = "sample_invoice.png"
    
    if not os.path.exists(invoice_path):
        print(f"❌ Sample invoice not found: {invoice_path}")
        return
    
    models_to_test = ["ollama", "lmstudio"]
    
    for model_type in models_to_test:
        print(f"\n🤖 Testing {model_type.upper()}...")
        
        try:
            result = ocr.extract_invoice_data(invoice_path, model_type=model_type)
            
            if result:
                print(f"   ✅ {model_type.upper()} - Success")
                print(f"   📊 Vendor: {result.get('vendor_name', 'Unknown')}")
                print(f"   📊 Items: {len(result.get('items', []))}")
                print(f"   📊 Total: ${result.get('total', 0)}")
                
                # Save model-specific result
                output_file = f"result_{model_type}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"   💾 Saved to: {output_file}")
            else:
                print(f"   ❌ {model_type.upper()} - Failed")
                
        except Exception as e:
            print(f"   ❌ {model_type.upper()} - Error: {e}")

def example_custom_processing():
    """Example with custom processing steps"""
    print("\n🔹 Custom Processing Example")
    print("-" * 30)
    
    ocr = EnhancedInvoiceOCR(debug=True)
    
    invoice_path = "sample_invoice.png"
    
    if not os.path.exists(invoice_path):
        print(f"❌ Sample invoice not found: {invoice_path}")
        return
    
    # Step 1: Get processed images for inspection
    print("🖼️  Step 1: Image preprocessing...")
    processed_images = ocr.advanced_image_preprocessing(invoice_path)
    print(f"   Generated {len(processed_images)} image variants")
    
    # Save debug images
    ocr.save_debug_images(processed_images, "debug_output")
    print("   💾 Debug images saved to: debug_output/")
    
    # Step 2: Extract text with ensemble method
    print("\n📝 Step 2: Text extraction...")
    extracted_text = ocr.extract_text_ensemble(invoice_path)
    
    if extracted_text:
        print(f"   ✅ Extracted {len(extracted_text)} characters")
        
        # Step 3: Clean and enhance text
        print("\n🧹 Step 3: Text cleaning...")
        cleaned_text = ocr.clean_and_enhance_text(extracted_text)
        print(f"   ✅ Cleaned text: {len(cleaned_text)} characters")
        
        # Save intermediate results
        with open("raw_text.txt", 'w') as f:
            f.write(extracted_text)
        with open("cleaned_text.txt", 'w') as f:
            f.write(cleaned_text)
        
        print("   💾 Saved raw_text.txt and cleaned_text.txt")
        
        # Step 4: Process with LLM
        print("\n🤖 Step 4: LLM processing...")
        result = ocr.process_with_llm(cleaned_text, "ollama")
        
        if result:
            print("   ✅ LLM processing successful!")
            
            # Save final result
            with open("final_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            print("   💾 Saved final_result.json")
        else:
            print("   ❌ LLM processing failed")
    else:
        print("   ❌ Text extraction failed")

def main():
    """Run all examples"""
    print("🚀 Enhanced Invoice OCR - Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Text-Only Extraction", example_text_only_extraction),
        ("Different LLM Models", example_different_models),
        ("Custom Processing", example_custom_processing),
        ("Batch Processing", example_batch_processing),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nChoose an example to run:")
    print("0. Run all examples")
    print("q. Quit")
    
    while True:
        try:
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == '0':
                print("\n🔄 Running all examples...")
                for name, func in examples:
                    print(f"\n{'='*20} {name} {'='*20}")
                    func()
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    name, func = examples[idx]
                    print(f"\n{'='*20} {name} {'='*20}")
                    func()
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()