import re
import warnings
from typing import Any, Dict

import torch

# 1. Apply Bypass BEFORE importing transformers
import app.services.hf_bypass  # type: ignore

# Try importing IPEX for Intel GPU acceleration
try:
    import intel_extension_for_pytorch as ipex  # type: ignore

    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

# Suppress warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)


class FlorenceService:
    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        print(f"--- Loading {model_name} ---")

        # 1. Device Selection Logic
        if HAS_IPEX and hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.dtype = torch.bfloat16
            print(">>> Using Intel XPU (GPU)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            print(">>> Using NVIDIA CUDA")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print(">>> Using CPU (Slow fallback)")

        # 2. Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",  # Fast Native Attention
            torch_dtype=self.dtype,
        ).to(self.device)

        # 3. Optimize for Intel (if applicable)
        if HAS_IPEX:
            print(">>> Optimizing model with IPEX...")
            self.model = ipex.optimize(self.model, dtype=self.dtype)

        # 4. Load Processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        print("--- Model Loaded Successfully ---")

    def predict(
        self, image: Image.Image, task: str = "<OCR_WITH_REGION>"
    ) -> Dict[str, Any]:
        """Run inference on a PIL Image."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(text=task, images=image, return_tensors="pt").to(
            self.device
        )

        # Cast inputs to correct dtype if using XPU/CUDA
        if self.device != "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,  # GREEDY SEARCH (Fastest)
            do_sample=False,
            early_stopping=False,
        )

        result_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse result using the logic we built earlier
        return self._parse_card_layout(result_text)

    def _parse_card_layout(self, result: str) -> Dict[str, str]:
        """Parses raw Florence output into ID, Title, and Description."""
        result = result.replace("</s>", "").replace("<s>", "").strip()

        pattern = r"([^<]+)((?:<loc_\d+>)+)"
        matches = re.finditer(pattern, result)

        card_data = {"id": [], "title": [], "description": []}

        for match in matches:
            text = match.group(1).strip()
            loc_string = match.group(2)
            if not text:
                continue

            coords = [int(n) for n in re.findall(r"<loc_(\d+)>", loc_string)]
            if len(coords) < 4:
                continue

            # Calculate Center Y and Center X
            y_coords = coords[1::2]
            x_coords = coords[0::2]
            y_center = sum(y_coords) / len(y_coords)
            x_center = sum(x_coords) / len(x_coords)

            # --- SPATIAL LOGIC ---
            if y_center < 350:
                if x_center < 500:
                    card_data["id"].append(text)
                else:
                    card_data["title"].append(text)
            elif y_center > 600:
                card_data["description"].append(text)

        return {
            "id": " ".join(card_data["id"]).replace(
                " ", ""
            ),  # Remove spaces in ID like "n 1" -> "n1"
            "title": " ".join(card_data["title"]),
            "description": " ".join(card_data["description"]),
        }
