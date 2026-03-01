import io
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.schemas import OCRResponse

# Import our new modules
from app.services.florence import FlorenceService

# Global model instance
model_instance: FlorenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global model_instance
    try:
        model_instance = FlorenceService()
        yield
    finally:
        model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="La Fallera OCR API", lifespan=lifespan)


@app.get("/health")
def health():
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model initializing")
    return {"status": "ready", "device": model_instance.device}


@app.post("/extract", response_model=OCRResponse)
async def extract_text(file: Annotated[UploadFile, File(...)]):
    """
    Upload a card image
    Get extracted text data from the card
    """
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
