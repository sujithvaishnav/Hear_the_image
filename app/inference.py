from PIL import Image
from utils.vision import generate_caption, run_ocr, synthesize_speech

def run_pipeline(
    image: Image.Image,
    lang_code: str = "en",
    include_ocr: bool = False
):
    caption = generate_caption(image)

    ocr_text = None
    if include_ocr:
        ocr_text = run_ocr(image, ocr_langs=["en"])

    parts = [caption]
    if ocr_text:
        parts.append(f"I can read some text: {ocr_text}.")
    final_text = " ".join(parts).strip()

    if not final_text:
        final_text = (
            "I could not confidently describe the image. "
            "Please try another angle or brighter lighting."
        )

    audio_bytes = synthesize_speech(final_text, lang_code)

    return {
        "caption": caption,
        "full_text": final_text,
        "audio_bytes": audio_bytes
    }
