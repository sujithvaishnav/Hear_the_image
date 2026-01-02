from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from app.inference import run_pipeline
from app.schemas import PredictResponse

app = FastAPI(title="HearTheImage API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    include_ocr: bool = Form(False),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = run_pipeline(
        image=image,
        lang_code=lang,
        include_ocr=include_ocr
    )

    return {
        "caption": result["caption"],
        "text": result["full_text"],
    }

@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    include_ocr: bool = Form(False),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = run_pipeline(
        image=image,
        lang_code=lang,
        include_ocr=include_ocr
    )

    return StreamingResponse(
        io.BytesIO(result["audio_bytes"]),
        media_type="audio/mpeg"
    )

