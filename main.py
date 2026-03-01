from poc.florence import FlorenceOCR
from poc.tesseract import POCTesseract

tesseract_ocr = POCTesseract()
florence_ocr = FlorenceOCR()

# image_path = "poc/output/01_normalized.png"
image_path = "poc/output/05_ocr_ready.png"
# text = tesseract_ocr.get_image_text(image_path=image_path, language="cat")
# print("Extracted text:")
# print(text)
text = florence_ocr.extract_text(image_path, task="<OCR_WITH_REGION>")
print(text["parsed"])
# print(text)
