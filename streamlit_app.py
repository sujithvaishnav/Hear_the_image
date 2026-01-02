import time
import streamlit as st
from PIL import Image, ImageOps

import requests
import io

try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Image → Audio Scene Describer",
    page_icon="🦮",
    layout="centered"
)

st.title("🦮 Image → Audio Scene Describer")
st.caption("Accessible AI: describe what’s in a photo and speak it out.")

# -------------------------
# Input Section
# -------------------------
col1, col2 = st.columns([1, 1])

with col1:
    source = st.radio(
        "Select image source",
        ["Upload", "Camera"],
        index=0,
        horizontal=True
    )

    if source == "Upload":
        file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False
        )
        img = Image.open(file).convert("RGB") if file else None
    else:
        cam = st.camera_input("Take a photo")
        img = Image.open(cam).convert("RGB") if cam else None

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
        help=(
            "If enabled, the app reads printed text in the image. Requires EasyOCR. "
            + ("✅ Detected." if OCR_AVAILABLE else "❌ Not installed")
        ),
    )

st.divider()

# -------------------------
# Preview
# -------------------------
if img is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

preview = ImageOps.contain(img, (768, 768))
st.image(preview, caption="Input image", use_container_width=True)

# -------------------------
# Inference
# -------------------------
if st.button("🔊 Describe & Speak", type="primary"):
    with st.spinner("Analyzing image..."):
        try:
            t0 = time.time()
            # Convert PIL Image → raw bytes ONCE
            img_io = io.BytesIO()
            img.save(img_io, format="JPEG")
            img_bytes = img_io.getvalue()

            data = {
                "lang": lang,
                "include_ocr": add_ocr
            }

            # ---- Caption API ----
            files = {
                "file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")
            }
            resp = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files,
                data=data,
                timeout=90
            )
            resp.raise_for_status()
            caption = resp.json()["caption"]

            # ---- Audio API ----
            files = {
                "file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")
            }
            audio_resp = requests.post(
                "http://127.0.0.1:8000/predict/audio",
                files=files,
                data=data,
                timeout=90
            )
            audio_resp.raise_for_status()
            audio_bytes = audio_resp.content

            elapsed = time.time() - t0
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("Done!")
    st.write("**Description:**", caption)

    st.audio(audio_bytes, format="audio/mp3")
    st.caption(
        f"Processed in ~{elapsed:.1f}s "
        "(first run may take longer due to model downloads)."
    )

# -------------------------
# Notes & Future Work
# -------------------------
with st.expander("Design notes & future improvements"):
    st.markdown(
        "- Replace gTTS with a local neural TTS for offline use (e.g., Coqui TTS)\n"
        "- Add object grounding (e.g., YOLO) and spatial reasoning (left/right/near)\n"
        "- Add on-device quantization (TFLite / ONNX)\n"
        "- Add multi-turn voice interface with wake-word controls\n"
        "- Add haptic feedback patterns for accessibility use cases"
    )
