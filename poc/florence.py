import json
import re
from pathlib import Path
from typing import Any, Dict

# Import our custom patch to bypass flash_attn requirement
try:
    from . import hf_bypass  # Works when imported from main.py
except ImportError:
    import hf_bypass
import torch
from PIL import Image

# Now it is safe to import Hugging Face transformers
from transformers import (
    AutoModelForCausalLM,  # type: ignore
    AutoProcessor,  # type: ignore
)


class FlorenceOCR:
    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
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
            # 1. Regex to extract text and location tags
            # Matches: "Some Text" followed by "<loc_123><loc_456>..."
            pattern = r"([^<]+)((?:<loc_\d+>)+)"
            matches = re.finditer(pattern, result)

            card_data = {
                "card_number": [],
                "title": [],
                "body": [],
                "raw_text": [],  # Keep everything just in case
            }

            for match in matches:
                text = match.group(1).strip()
                loc_string = match.group(2)

                if not text:
                    continue

                # Extract coordinates
                # Format: <loc_x1><loc_y1><loc_x2><loc_y2>...
                coords = [int(n) for n in re.findall(r"<loc_(\d+)>", loc_string)]

                if len(coords) < 4:
                    continue

                # Calculate center points of the text box
                # X coordinates are at indices 0, 2, 4...
                # Y coordinates are at indices 1, 3, 5...
                x_coords = coords[0::2]
                y_coords = coords[1::2]

                x_center = sum(x_coords) / len(x_coords)
                y_center = sum(y_coords) / len(y_coords)

                # --- SPATIAL LOGIC FOR LA FALLERA CALAVERA ---
                # Florence coordinates are 0-1000

                # 1. HEADER ZONE (Top 35% of the card)
                if y_center < 350:
                    # Split Left vs Right
                    if x_center < 500:
                        card_data["card_number"].append(text)
                    else:
                        card_data["title"].append(text)

                # 2. BODY ZONE (Bottom 40% of the card)
                elif y_center > 600:
                    card_data["body"].append(text)

                # 3. ART ZONE (Middle) - Usually noise or artist name
                # We typically ignore this for gameplay rules, or add to body if unsure
                else:
                    # Optional: Add to body if it looks like rule text
                    pass

            # Join lists into clean strings
            structured_result = {
                "id": " ".join(card_data["card_number"]),
                "title": " ".join(card_data["title"]),
                "description": " ".join(card_data["body"]),
                "full_text": result,  # The raw string for debugging
            }

            # Post-processing cleanup
            # Sometimes 'n1' is read as 'n 1', fix that
            structured_result["id"] = structured_result["id"].replace(" ", "")

            return structured_result

        # JSON parsing for Caption tasks
        if task in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
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
