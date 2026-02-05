#!/usr/bin/env python3
"""
Installation and dependency test script for Enhanced Invoice OCR system.
Run this script to verify all components are working correctly.
"""

import sys
import os
import requests
import json
from typing import List, Tuple

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def test_dependencies():
    """Test required Python dependencies"""
    print("\n📦 Testing Python dependencies...")
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("pandas", "pandas"),
        ("requests", "requests"),
    ]
    
    optional_dependencies = [
        ("easyocr", "easyocr"),
        ("pytesseract", "pytesseract"),
        ("fitz", "PyMuPDF"),
    ]
    
    all_good = True
    
    # Test required dependencies
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {package} - OK")
        except ImportError:
            print(f"   ❌ {package} - MISSING (pip install {package})")
            all_good = False
    
    # Test optional dependencies
    for module, package in optional_dependencies:
        try:
            __import__(module)
            print(f"   ✅ {package} - OK")
        except ImportError:
            print(f"   ⚠️  {package} - OPTIONAL (pip install {package})")
    
    return all_good

def test_tesseract():
    """Test Tesseract OCR installation"""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract {version} - OK")
        return True
    except ImportError:
        print("   ⚠️  pytesseract not installed")
        return False
    except Exception as e:
        print(f"   ❌ Tesseract not found - {e}")
        print("   💡 Install: brew install tesseract (macOS) or sudo apt install tesseract-ocr (Ubuntu)")
        return False

def test_easyocr():
    """Test EasyOCR functionality"""
    print("\n👁️  Testing EasyOCR...")
    
    try:
        import easyocr
        # Try to initialize reader (this downloads models on first run)
        print("   📥 Initializing EasyOCR (may download models)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("   ✅ EasyOCR - OK")
        return True
    except ImportError:
        print("   ⚠️  EasyOCR not installed (pip install easyocr)")
        return False
    except Exception as e:
        print(f"   ❌ EasyOCR initialization failed - {e}")
        return False

def test_ollama_connection():
    """Test Ollama API connection"""
    print("\n🦙 Testing Ollama connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"   ✅ Ollama connected - {len(models)} models available:")
                for model in models[:3]:  # Show first 3 models
                    print(f"      - {model['name']}")
                return True
            else:
                print("   ⚠️  Ollama connected but no models found")
                print("   💡 Run: ollama pull llama3.2:latest")
                return False
        else:
            print(f"   ❌ Ollama API error - Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama not running")
        print("   💡 Start Ollama and run: ollama pull llama3.2:latest")
        return False
    except Exception as e:
        print(f"   ❌ Ollama connection failed - {e}")
        return False

def test_lmstudio_connection():
    """Test LM Studio API connection"""
    print("\n🏭 Testing LM Studio connection...")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            if models:
                print(f"   ✅ LM Studio connected - {len(models)} models available:")
                for model in models[:3]:  # Show first 3 models
                    print(f"      - {model['id']}")
                return True
            else:
                print("   ⚠️  LM Studio connected but no models loaded")
                return False
        else:
            print(f"   ❌ LM Studio API error - Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ LM Studio not running")
        print("   💡 Start LM Studio and load a model")
        return False
    except Exception as e:
        print(f"   ❌ LM Studio connection failed - {e}")
        return False

def test_image_processing():
    """Test basic image processing functionality"""
    print("\n🖼️  Testing image processing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        print("   ✅ Image processing - OK")
        return True
    except Exception as e:
        print(f"   ❌ Image processing failed - {e}")
        return False

def create_sample_invoice():
    """Create a sample invoice image for testing"""
    print("\n📄 Creating sample invoice for testing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a sample invoice image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add invoice content
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "ACME SERVICES INC", (50, 50), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "Date: 2024-01-15", (50, 100), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Service Description    $150.00", (50, 200), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "Additional Service     $75.00", (50, 250), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "Subtotal: $225.00", (50, 350), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Tax: $18.00", (50, 400), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Total: $243.00", (50, 450), font, 0.8, (0, 0, 0), 2)
        
        # Save sample invoice
        cv2.imwrite("sample_invoice.png", img)
        print("   ✅ Sample invoice created: sample_invoice.png")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create sample invoice - {e}")
        return False

def run_quick_test():
    """Run a quick test with the sample invoice"""
    print("\n🧪 Running quick extraction test...")
    
    if not os.path.exists("sample_invoice.png"):
        print("   ❌ Sample invoice not found")
        return False
    
    try:
        # Import and test the main OCR class
        from enhanced_invoice_ocr import EnhancedInvoiceOCR
        
        ocr = EnhancedInvoiceOCR(debug=False)
        
        # Test text extraction only (no LLM)
        text = ocr.extract_text_ensemble("sample_invoice.png")
        
        if text and len(text.strip()) > 10:
            print("   ✅ Text extraction - OK")
            print(f"   📝 Extracted: {len(text)} characters")
            return True
        else:
            print("   ❌ Text extraction failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Quick test failed - {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Enhanced Invoice OCR - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Tesseract OCR", test_tesseract),
        ("EasyOCR", test_easyocr),
        ("Image Processing", test_image_processing),
        ("Sample Invoice", create_sample_invoice),
        ("Quick Test", run_quick_test),
        ("Ollama API", test_ollama_connection),
        ("LM Studio API", test_lmstudio_connection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python enhanced_invoice_ocr.py sample_invoice.png")
        print("2. Try with your own invoice images")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        if not any(name == "Ollama API" and result for name, result in results) and \
           not any(name == "LM Studio API" and result for name, result in results):
            print("\n💡 No LLM backend available. Install either:")
            print("   - Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   - LM Studio: https://lmstudio.ai/")

if __name__ == "__main__":
    main()