from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import shutil
import uuid

from ocr_llm import FocusedInvoiceOCR


app = FastAPI()

# Upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed formats
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}


def allowed_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


@app.post("/upload-invoice/")
async def upload_invoice(file: UploadFile = File(...)):

    # 1. Validate file
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, PDF files are allowed"
        )

    # 2. Generate safe filename
    ext = os.path.splitext(file.filename)[1]
    new_name = f"{uuid.uuid4()}{ext}"

    file_path = os.path.join(UPLOAD_DIR, new_name)

    # 3. Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 4. Initialize OCR
        ocr = FocusedInvoiceOCR()

        # 5. Run your pipeline
        result = ocr.extract_invoice_data_hybrid(file_path)

        if not result:
            raise Exception("OCR processing failed")

        return {
            "status": "success",
            "original_file": file.filename,
            "stored_file": new_name,
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR Error: {str(e)}"
        )
