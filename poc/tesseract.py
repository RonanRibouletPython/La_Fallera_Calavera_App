import pytesseract
from PIL import Image


class POCTesseract:
    def __init__(self) -> None:
        self.language_list = ["en", "cat", "es"]

    def get_image_text(self, image_path: str, language: str = "en") -> str:
        try:
            # Check if language is valid (fixed: was using 'set' incorrectly)
            if language not in self.language_list:
                raise ValueError(
                    f"Language '{language}' not supported. Choose from: {self.language_list}"
                )

            # Open and process image
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image=image, lang=language)

            return text.strip()

        except FileNotFoundError:
            print(f"Error: Image file not found: {image_path}")
            raise
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in PATH")
            raise
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            raise
