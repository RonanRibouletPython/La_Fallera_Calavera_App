from poc.tesseract import POCTesseract

ocr = POCTesseract()

image_path = "images/ausias.png"
text = ocr.get_image_text(image_path=image_path, language="cat")
print("Extracted text:")
print(text)
