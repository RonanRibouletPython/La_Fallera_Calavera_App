import asyncio
import io
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.schemas import OCRResponse

# Import our new modules
from app.services.florence import FlorenceService

# Global model instance
model_instance: FlorenceService | None = None


class ModelState(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


model_state = ModelState.LOADING
model_instance: FlorenceService | None = None
model_error: str | None = None


async def load_model_async():
    """Load model in background without blocking startup"""
    global model_instance, model_state, model_error
    try:
        # Run CPU-bound model loading in thread pool
        loop = asyncio.get_event_loop()
        model_instance = await loop.run_in_executor(None, FlorenceService)
        model_state = ModelState.READY
        print("Model ready for inference")
    except Exception as e:
        model_state = ModelState.ERROR
        model_error = str(e)
        print(f"Model loading failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Non-blocking startup"""
    # Start loading in background
    asyncio.create_task(load_model_async())
    yield
    # Cleanup
    global model_instance
    model_instance = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="La Fallera OCR API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": model_state.value,
        "device": model_instance.device if model_instance else None,
        "error": model_error if model_state == ModelState.ERROR else None,
    }


@app.post("/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Upload a card image
    Get extracted text data from the card
    """
    if model_state == ModelState.LOADING:
        raise HTTPException(503, "Model still loading, try again in 30s")
    if model_state == ModelState.ERROR:
        raise HTTPException(500, f"Model failed to load: {model_error}")

    if not model_instance:
        raise HTTPException(status_code=503, detail="Model initializing")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, detail="Invalid file type. Use JPEG, PNG, or WebP.")

    # Read bytes
    contents = await file.read()

    try:
        # Convert to PIL
        image = Image.open(io.BytesIO(contents))

        # Run Inference in threadpool to avoid blocking async loop
        result = await run_in_threadpool(model_instance.predict, image=image)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
