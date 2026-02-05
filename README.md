# Enhanced Invoice OCR + LLM System

A powerful OCR+LLM system specifically optimized for extracting structured data from blurry, poorly captured, and low-quality invoice images.

## 🚀 Key Features

### Advanced Image Preprocessing
- **Multi-stage blur reduction** using unsharp masking and deconvolution
- **Adaptive noise reduction** with non-local means denoising
- **Dynamic contrast enhancement** using CLAHE
- **Multiple thresholding approaches** (Adaptive + Otsu)
- **Automatic image scaling** for optimal OCR performance

### Dual OCR Engine Support
- **EasyOCR**: Better for handwritten and stylized text
- **Tesseract**: Better for standard printed text
- **Ensemble approach**: Automatically selects best result

### Intelligent Text Processing
- **Smart line reconstruction** from OCR bounding boxes
- **OCR error correction** (fixes common mistakes like "8" → "$", "S" → "$")
- **Quality-based text selection** using multiple scoring metrics
- **Outlier detection** to filter corrupted text

### LLM-Powered Extraction
- **Structured data extraction** with precise JSON output
- **Multiple LLM backends**: Ollama and LM Studio support
- **Robust JSON parsing** with automatic error correction
- **Invoice-specific prompting** for accurate field extraction

## 📋 Extracted Data Fields

The system extracts the following structured data:

```json
{
  "vendor_name": "Company Name",
  "date": "2024-01-15",
  "subtotal": 1500.00,
  "tax_amount": 120.00,
  "discount_amount": 50.00,
  "total": 1570.00,
  "items": [
    {
      "item_name": "Service Description",
      "unit_price": 100.00,
      "quantity": 2,
      "amount": 200.00
    }
  ]
}
```

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### macOS:
```bash
brew install tesseract
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### Windows:
Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Setup LLM Backend

Choose one of the following:

#### Option A: Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama3.2:latest      # 2B model (fast, good quality)
ollama pull llama3.2:3b          # 3B model (slower, better quality)
ollama pull qwen2.5:3b           # Alternative 3B model
```

#### Option B: LM Studio
1. Download LM Studio from: https://lmstudio.ai/
2. Install and run LM Studio
3. Download a model (e.g., "openai/gpt-oss-20b")
4. Start the local server on port 1234

## 🚀 Usage

### Basic Usage

```bash
# Process a single invoice
python enhanced_invoice_ocr.py invoice.jpg

# Use LM Studio instead of Ollama
python enhanced_invoice_ocr.py invoice.pdf --model lmstudio

# Enable debug output
python enhanced_invoice_ocr.py invoice.png --debug
```

### Interactive Mode

If you don't specify a file, the system will show available files in the current directory:

```bash
python enhanced_invoice_ocr.py
```

### Python API Usage

```python
from enhanced_invoice_ocr import EnhancedInvoiceOCR

# Initialize with debug mode
ocr = EnhancedInvoiceOCR(debug=True)

# Extract data from invoice
result = ocr.extract_invoice_data("path/to/invoice.jpg", model_type="ollama")

if result:
    print(f"Vendor: {result['vendor_name']}")
    print(f"Total: ${result['total']}")
    print(f"Items: {len(result['items'])}")
else:
    print("Extraction failed")
```

## 🎯 Optimization for Blurry Images

The system is specifically optimized for challenging image conditions:

### Image Quality Issues Handled:
- ✅ **Blurry/out-of-focus images**
- ✅ **Poor lighting conditions**
- ✅ **Low resolution images**
- ✅ **Skewed/rotated documents**
- ✅ **Noisy backgrounds**
- ✅ **Faded or low contrast text**

### Preprocessing Pipeline:
1. **Automatic scaling** to optimal resolution
2. **Blur reduction** using multiple techniques
3. **Noise removal** with advanced filtering
4. **Contrast enhancement** for better text visibility
5. **Multiple thresholding** approaches tested
6. **Best result selection** based on quality metrics

## 🔧 Configuration

### OCR Confidence Thresholds

Adjust in the `__init__` method:

```python
self.easyocr_threshold = 0.3    # Lower = more text, higher noise
self.tesseract_threshold = 30   # Tesseract confidence (0-100)
```

### LLM Parameters

Modify in the respective processing methods:

```python
# Ollama settings
"temperature": 0.1,     # Lower = more consistent
"top_p": 0.9,          # Nucleus sampling
"num_predict": 1500,   # Max tokens

# LM Studio settings
"temperature": 0.1,
"max_tokens": 1500
```

## 📊 Performance Tips

### For Best Results:
1. **Image Quality**: Use highest resolution available
2. **Lighting**: Ensure even lighting across the document
3. **Orientation**: Keep document as straight as possible
4. **Format**: PNG or high-quality JPEG preferred
5. **Model Choice**: Use larger LLM models for complex invoices

### Speed Optimization:
- Use smaller LLM models (llama3.2:latest) for faster processing
- Disable debug mode in production
- Process multiple images in batch

## 🐛 Troubleshooting

### Common Issues:

#### "No text extracted"
- Check image quality and format
- Try different preprocessing variants
- Ensure Tesseract is properly installed

#### "LLM API connection failed"
- Verify Ollama/LM Studio is running
- Check API endpoints (localhost:11434 for Ollama, localhost:1234 for LM Studio)
- Ensure model is loaded

#### "JSON parsing error"
- Try a different LLM model
- Check if model supports instruction following
- Enable debug mode to see raw LLM output

#### "Poor extraction quality"
- Use higher resolution images
- Try different LLM models
- Adjust OCR confidence thresholds

### Debug Mode

Enable debug mode to see detailed processing information:

```bash
python enhanced_invoice_ocr.py invoice.jpg --debug
```

This will show:
- Image preprocessing steps
- OCR engine results and scores
- Text quality metrics
- LLM prompt and response
- JSON parsing attempts

## 📁 Output Files

The system generates:
- `{filename}_extracted.json`: Structured invoice data
- `debug_images/`: Preprocessed images (debug mode only)

## 🔄 Supported File Formats

- **Images**: JPG, JPEG, PNG, TIFF, BMP
- **Documents**: PDF (text extraction + OCR fallback)

## 🎯 Use Cases

Perfect for:
- **Accounting automation**
- **Expense management systems**
- **Document digitization**
- **Financial data extraction**
- **Invoice processing workflows**

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.