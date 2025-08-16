from models import load_blip
from typing import Optional
from PIL import Image, ImageOps
from gtts import gTTS
import io

try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

def get_ocr_reader(lang_list):
    if not OCR_AVAILABLE:
        return None
    return easyocr.Reader(lang_list, gpu=False)

def generate_caption(img: Image.Image) -> str:
    from transformers import BlipProcessor
    processor, model = load_blip()
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def run_ocr(img: Image.Image, ocr_langs: list[str]) -> Optional[str]:
    if not OCR_AVAILABLE:
        return None
    reader = get_ocr_reader(ocr_langs)
    result = reader.readtext(np_image(img), detail=0)
    text = ". ".join([t.strip() for t in result if t and t.strip()])
    return text if text else None


def np_image(img: Image.Image):
    import numpy as np
    return np.array(img.convert("RGB"))


def synthesize_speech(text: str, lang_code: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang_code)
    bio = io.BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio.read()