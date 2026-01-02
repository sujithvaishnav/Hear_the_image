from typing import Optional
from PIL import Image
from gtts import gTTS
import io

from models.load_blip import load_blip

try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -----------------------------
# Global caches (backend style)
# -----------------------------
_blip_cache = {}
_ocr_cache = {}

# -----------------------------
# Captioning
# -----------------------------
def generate_caption(img: Image.Image) -> str:
    from transformers import BlipProcessor

    if "model" not in _blip_cache:
        processor, model = load_blip()
        _blip_cache["processor"] = processor
        _blip_cache["model"] = model
    else:
        processor = _blip_cache["processor"]
        model = _blip_cache["model"]

    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

# -----------------------------
# OCR
# -----------------------------
def get_ocr_reader(lang_list):
    if not OCR_AVAILABLE:
        return None

    key = tuple(lang_list)
    if key not in _ocr_cache:
        _ocr_cache[key] = easyocr.Reader(lang_list, gpu=False)

    return _ocr_cache[key]

def run_ocr(img: Image.Image, ocr_langs: list[str]) -> Optional[str]:
    if not OCR_AVAILABLE:
        return None

    reader = get_ocr_reader(ocr_langs)
    if reader is None:
        return None

    result = reader.readtext(np_image(img), detail=0)
    text = ". ".join([t.strip() for t in result if t and t.strip()])
    return text if text else None

def np_image(img: Image.Image):
    import numpy as np
    return np.array(img.convert("RGB"))

# -----------------------------
# Text-to-Speech
# -----------------------------
def synthesize_speech(text: str, lang_code: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang_code)
    bio = io.BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio.read()
