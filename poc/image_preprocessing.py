import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessing:
    def __init__(self, image_path: Optional[str] = None):
        self.image_path = image_path
        self.original_image = None
        self.current_image = None

        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path: str) -> np.ndarray:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.original_image = image.copy()
        self.current_image = image.copy()
        self.image_path = image_path

        return self.current_image

    def normalize(
        self, alpha: int = 0, beta: int = 255, norm_type: int = cv2.NORM_MINMAX
    ) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        normalized_img = np.zeros(
            self.current_image.shape, dtype=self.current_image.dtype
        )

        normalized_img = cv2.normalize(
            self.current_image,
            normalized_img,
            alpha=alpha,
            beta=beta,
            norm_type=norm_type,
        )

        self.current_image = normalized_img
        print("Normalized image")
        return normalized_img

    def deskew(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        img = self.current_image.copy()

        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        self.current_image = rotated
        print(f"Deskewed image by {angle:.2f} degrees")

        return rotated

    def get_greyscale(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image = gray
        print("Converted to grayscale")

        return gray

    def thresholding(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        _, binary = cv2.threshold(
            self.current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        self.current_image = binary
        print("Applied binary thresholding")

        return binary

    def denoise(self, h: int = 10) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        if len(self.current_image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                self.current_image, None, h, h, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(self.current_image, None, h, 7, 21)

        self.current_image = denoised
        print("Denoised image")

        return denoised

    def remove_noise(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(self.current_image, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.medianBlur(img, 3)

        self.current_image = img
        print("Removed noise")

        return img

    def thin_font(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        img = cv2.bitwise_not(self.current_image)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.bitwise_not(img)

        self.current_image = img
        print("Thinned font")

        return img

    def thick_font(self) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        img = cv2.bitwise_not(self.current_image)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.bitwise_not(img)

        self.current_image = img
        print("Thickened font")

        return img

    def set_image_dpi(self, target_dpi: int = 300, max_width: int = 1024) -> str:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        if len(self.current_image.shape) == 3:
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = Image.fromarray(self.current_image)

        width, height = pil_image.size
        factor = min(1, float(max_width) / width)
        new_width = int(factor * width)
        new_height = int(factor * height)

        if factor < 1:
            im_resized = pil_image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized from {width}x{height} to {new_width}x{new_height}")
        else:
            im_resized = pil_image
            print(f"No resize needed ({width}x{height})")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_filename = temp_file.name
        temp_file.close()

        im_resized.save(temp_filename, dpi=(target_dpi, target_dpi))
        print(f"Saved with DPI={target_dpi} to: {temp_filename}")

        return temp_filename

    def optimize_for_ocr(
        self,
        target_dpi: int = 300,
        max_width: int = 1024,
        output_path: Optional[str] = None,
    ) -> str:
        if self.current_image is None:
            raise ValueError("No image loaded. Use load_image() first.")

        if len(self.current_image.shape) == 3:
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = Image.fromarray(self.current_image)

        width, height = pil_image.size
        factor = min(1, float(max_width) / width)
        new_size = (int(factor * width), int(factor * height))

        if factor < 1:
            im_resized = pil_image.resize(new_size, Image.LANCZOS)
        else:
            im_resized = pil_image

        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            output_path = temp_file.name
            temp_file.close()
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        im_resized.save(output_path, dpi=(target_dpi, target_dpi))
        print(f"✓ Optimized: {new_size[0]}x{new_size[1]} @ {target_dpi} DPI")
        print(f"✓ Saved to: {output_path}")

        return output_path

    def save_image(self, output_path: str, image: Optional[np.ndarray] = None) -> None:
        img_to_save = image if image is not None else self.current_image

        if img_to_save is None:
            raise ValueError("No image to save")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(output_path, img_to_save)
        print(f"✓ Saved image to: {output_path}")

    def reset(self) -> None:
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            print("✓ Reset to original image")

    def get_image_info(self) -> dict:
        if self.current_image is None:
            return {"status": "No image loaded"}

        return {
            "shape": self.current_image.shape,
            "dtype": str(self.current_image.dtype),
            "min_value": self.current_image.min(),
            "max_value": self.current_image.max(),
            "mean_value": self.current_image.mean(),
        }


if __name__ == "__main__":
    preprocessor = ImagePreprocessing()

    image_path = "../images/carta-web-abradelo.png"
    preprocessor.load_image(image_path)

    print("Original image info:")
    print(preprocessor.get_image_info())

    print("\n=== Preprocessing Pipeline ===\n")

    preprocessor.normalize(alpha=0, beta=255)
    preprocessor.save_image("output/01_normalized.png")

    preprocessor.get_greyscale()
    preprocessor.save_image("output/02_greyscale.png")

    preprocessor.denoise(h=10)
    preprocessor.save_image("output/03_denoised.png")

    preprocessor.thresholding()
    preprocessor.save_image("output/04_binary.png")

    ocr_ready = preprocessor.optimize_for_ocr(
        target_dpi=300, max_width=1024, output_path="output/05_ocr_ready.png"
    )

    print("\n✓ All preprocessing steps completed!")
    print(f"✓ OCR-ready image: {ocr_ready}")
