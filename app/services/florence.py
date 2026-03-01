# app/services/florence.py
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

# Apply Bypass before importing transformers
import app.services.hf_bypass  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)


class FlorenceService:
    """
    Florence-2 Vision Language Model service optimized for card OCR.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        cache_dir: Path = Path.home() / ".cache" / "florence-ocr",
    ):
        logger.info(f"Initializing FlorenceService with {model_name}")

        self.model_name = model_name
        self.cache_dir = cache_dir

        # Setup device and optimization strategy
        self._setup_device()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model and processor
        self._load_model()
        self._load_processor()

        # Apply optimizations
        self._optimize_model()

        # Set to evaluation mode (disables dropout, etc.)
        self.model.eval()

        logger.info(f"Model ready on {self.device}")

    def _setup_device(self):
        """Configure compute device for CPU"""
        self.device = "cpu"
        self.dtype = torch.float32
        logger.info("Using CPU")

        # Optimize CPU threading
        num_threads = min(torch.get_num_threads(), 4)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
        logger.info(f"CPU threads: {num_threads} intra, 1 inter")

    def _load_model(self):
        """Load Florence-2 model with caching"""
        try:
            logger.info(f"Loading model from cache: {self.cache_dir}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation="sdpa",  # Use PyTorch's optimized attention
                torch_dtype=self.dtype,
                cache_dir=str(self.cache_dir),
                local_files_only=False,
            ).to(self.device)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _load_processor(self):
        """Load image processor and tokenizer"""
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
            )
            logger.info("Processor loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise RuntimeError(f"Processor loading failed: {e}")

    def _optimize_model(self):
        """Apply device-specific optimizations"""

        # Use PyTorch's native optimizations
        if self.device == "cpu":
            # Enable MKL-DNN optimizations if available
            if (
                hasattr(torch.backends, "mkldnn")
                and torch.backends.mkldnn.is_available()
            ):
                torch.backends.mkldnn.enabled = True
                logger.info("MKL-DNN optimizations enabled")

            # Enable oneDNN graph optimizations
            if hasattr(torch.jit, "set_fusion_strategy"):
                torch.jit.set_fusion_strategy([("STATIC", 3)])
                logger.info("JIT fusion optimizations enabled")

    @torch.inference_mode()
    def predict(
        self, image: Image.Image, task: str = "<OCR_WITH_REGION>"
    ) -> Dict[str, Any]:
        """
        Extract text from a card image

        Args:
            image: PIL Image of a game card
            task: Florence-2 task prompt

        Returns:
            Dict with keys: id, title, description
        """
        try:
            # Preprocess image
            image = self._preprocess_image(image)

            # Prepare inputs
            inputs = self.processor(text=task, images=image, return_tensors="pt").to(
                self.device
            )

            # Cast to appropriate dtype for GPU
            if self.device != "cpu":
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            # Generate text
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,  # Greedy search (fastest)
                do_sample=False,
                early_stopping=False,
            )

            # Decode result
            result_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Parse into structured format
            return self._parse_card_layout(result_text)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to process image: {e}")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess user-taken photos for better OCR"""
        # Ensure RGB mode
        if image.mode != "RGB":
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")

        # Resize if extremely large
        max_dimension = 1280
        width, height = image.size

        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            logger.debug(f"Resizing image: {image.size} -> {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Warn on very small images
        min_dimension = 224
        if min(width, height) < min_dimension:
            logger.warning(f"Image very small: {image.size}, may affect accuracy")

        return image

    def _parse_card_layout(self, result: str) -> Dict[str, str]:
        """Parse Florence-2 OCR output into structured card data"""
        # Clean special tokens
        result = result.replace("</s>", "").replace("<s>", "").strip()

        # Regex: capture text + bounding box coordinates
        pattern = r"([^<]+)((?:<loc_\d+>)+)"
        matches = re.finditer(pattern, result)

        card_data = {"id": [], "title": [], "description": []}

        for match in matches:
            text = match.group(1).strip()
            loc_string = match.group(2)

            if not text:
                continue

            # Extract coordinates
            coords = [int(n) for n in re.findall(r"<loc_(\d+)>", loc_string)]

            if len(coords) < 4:
                logger.debug(f"Skipping text with incomplete coords: {text}")
                continue

            # Calculate center position
            y_coords = coords[1::2]
            x_coords = coords[0::2]

            y_center = sum(y_coords) / len(y_coords)
            x_center = sum(x_coords) / len(x_coords)

            # Spatial classification
            if y_center < 350:  # Top half
                if x_center < 500:  # Left
                    card_data["id"].append(text)
                else:  # Right
                    card_data["title"].append(text)
            elif y_center > 600:  # Bottom
                card_data["description"].append(text)

        return {
            "id": " ".join(card_data["id"]),
            "title": " ".join(card_data["title"]),
            "description": " ".join(card_data["description"]),
        }
