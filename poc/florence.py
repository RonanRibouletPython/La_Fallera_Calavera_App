import json
import re
from pathlib import Path
from typing import Any, Dict

# Import our custom patch to bypass flash_attn requirement
import hf_bypass  # type: ignore
import torch
from PIL import Image

# Now it is safe to import Hugging Face transformers
from transformers import AutoModelForCausalLM, AutoProcessor


class FlorenceOCR:
    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        """Initialize Florence-2 model for OCR"""
        print(f"Loading {model_name}...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model using SDPA
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print("Model loaded successfully")

    def extract_text(self, image_path: str, task: str = "<OCR>") -> Dict[str, Any]:
        """Extract text from image using Florence-2."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(text=task, images=image, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            early_stopping=False,
        )

        result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[
            0
        ]

        parsed = self._parse_result(result, task)

        return {
            "task": task,
            "raw_output": result,
            "parsed": parsed,
            "image_path": image_path,
        }

    def _parse_result(self, result: str, task: str) -> Any:
        """Parse Florence-2 output based on task type."""
        result = result.replace("</s>", "").replace("<s>", "").strip()

        if task == "<OCR>":
            return result

        if task == "<OCR_WITH_REGION>":
            # Extract text by removing location coordinates
            # Format is: <loc_X><loc_Y>...text...<loc_X><loc_Y>...
            text_parts = []
            # Split by location tags and keep only the text parts
            parts = re.split(r"<loc_\d+>", result)
            for part in parts:
                part = part.strip()
                if part and not part.startswith("<") and len(part) > 1:
                    text_parts.append(part)

            return "\n".join(text_parts) if text_parts else result

        if task in [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
        ]:
            try:
                json_match = re.search(r"\{.*\}", result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return result


if __name__ == "__main__":
    print("=== Florence-2 OCR Test ===\n")

    ocr = FlorenceOCR()

    test_image = "output/05_ocr_ready.png"

    if not Path(test_image).exists():
        print(f"Error: {test_image} not found")
        print("Please update the path to your card image")
        exit(1)

    print(f"\n=== Extracting text from: {test_image} ===\n")

    result_basic = ocr.extract_text(test_image, task="<OCR_WITH_REGION>")
    print("Extracted text:")
    print(result_basic["parsed"])
    print()

    print("Test completed!")
