import os
import time
import streamlit as st
from PIL import Image, ImageOps
from models import load_blip
from utils import generate_caption,run_ocr,synthesize_speech

try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Image → Audio Scene Describer", page_icon="🦮", layout="centered")


st.title("🦮 Image → Audio Scene Describer")
st.caption("Accessible AI: describe what’s in a photo and speak it out.")

col1, col2 = st.columns([1, 1])
with col1:
    source = st.radio("Select image source", ["Upload", "Camera"], index=0, horizontal=True)
    if source == "Upload":
        file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
        img = Image.open(file).convert("RGB") if file else None
    else:
        img = st.camera_input("Take a photo")
        img = Image.open(img).convert("RGB") if img else None

with col2:
    st.markdown("**Output settings**")
    lang = st.selectbox(
        "Speech language",
        options=[
            ("English", "en"),
            ("Hindi", "hi"),
            ("Telugu", "te"),
            ("Tamil", "ta"),
            ("Kannada", "kn"),
        ],
        index=0,
        format_func=lambda x: x[0],
    )[1]
    add_ocr = st.toggle(
        "Include text in the scene (OCR)",
        value=False,
        help=("If enabled, the app reads printed text in the image. Requires EasyOCR. "
              + ("✅ Detected." if OCR_AVAILABLE else "❌ Not installed (pip install easyocr)")
        ),
    )

st.divider()

if img is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

preview = ImageOps.contain(img, (768, 768))
st.image(preview, caption="Input image", use_column_width=True)

if st.button("🔊 Describe & Speak", type="primary"):
    with st.spinner("Analyzing image..."):
        try:
            t0 = time.time()
            cap = generate_caption(img)
            ocr_text = None
            if add_ocr:
                ocr_text = run_ocr(img, ocr_langs=["en"])

            parts = [cap]
            if ocr_text:
                parts.append(f"I can read some text: {ocr_text}.")
            final_text = " ".join(parts)

            if not final_text.strip():
                final_text = "I could not confidently describe the image. Please try another angle or brighter lighting."

            audio_bytes = synthesize_speech(final_text, lang_code=lang)
            t_analyze = time.time() - t0

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("Done!")
    st.write("**Description:**", cap)
    if add_ocr and ocr_text:
        with st.expander("Detected text (OCR)", expanded=False):
            st.write(ocr_text)

    st.audio(audio_bytes, format="audio/mp3")
    st.caption(f"Processed in ~{t_analyze:.1f}s (first run may take longer due to model downloads).")


# -------------------------
# Notes & Future Work
# -------------------------
with st.expander("Design notes & future improvements"):
    st.markdown(
        "- Replace gTTS with a local neural TTS for offline use (e.g., Coqui TTS).\n"
        "- Add object grounding (e.g., YOLO) and spatial prepositions (left/right/near) for richer guidance.\n"
        "- Add on-device model quantization for edge devices (e.g., TFLite/ONNX).\n"
        "- Add multi-turn voice interface with wake word and repeat/slowdown controls.\n"
        "- Add haptic feedback patterns for critical objects (stairs, doors, vehicles)."
    )
