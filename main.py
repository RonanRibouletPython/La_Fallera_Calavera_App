from poc.tesseract import POCTesseract

ocr = POCTesseract()

# image_path = "poc/output/01_normalized.png"
image_path = "poc/output/05_ocr_ready.png"
text = ocr.get_image_text(image_path=image_path, language="cat")
print("Extracted text:")
print(text)
