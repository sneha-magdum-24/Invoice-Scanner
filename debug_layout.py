import easyocr
import json

def analyze_layout(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    
    # Sort by Y coordinate, then X
    # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    # we sort by top-left y, then top-left x
    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
    
    structured_data = []
    for (bbox, text, prob) in results:
        # Convert numpy types to native python for JSON serialization
        clean_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
        structured_data.append({
            "text": text,
            "bbox": clean_bbox,
            "y": int(bbox[0][1]) # top edge y
        })
        print(f"Y={int(bbox[0][1]):<4} | Text: {text}")

if __name__ == "__main__":
    analyze_layout("IMG_0183.jpg")
