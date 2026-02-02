import easyocr

def test_paragraph(image_path):
    reader = easyocr.Reader(['en'])
    # paragraph=True returns (bbox, text) or just [[bbox, text], ...] ? 
    # The return format changes with detail=0 or 1.
    # With paragraph=True, detail is ignored for the structure usually, but let's check.
    # standard readtext returns (bbox, text, prob).
    # paragraph=True returns [[bbox, text], ...] merging boxes.
    
    results = reader.readtext(image_path, paragraph=True)
    
    print("-" * 50)
    print("EXTRACTED TEXT WITH PARAGRAPH=TRUE")
    print("-" * 50)
    for res in results:
        # res is [bbox, text]
        text = res[1]
        print(text)

if __name__ == "__main__":
    test_paragraph("IMG_0182.jpg")
