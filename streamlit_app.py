import time
import io
import requests
import streamlit as st
from PIL import Image, ImageOps

# =========================
# CONFIG
# =========================
API_BASE_URL = "http://16.171.23.168:8000"  # EC2 Public IP

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Image → Audio Scene Describer",
    page_icon="🦮",
    layout="centered"
)

st.title("🦮 Image → Audio Scene Describer")
st.caption("Accessible AI: describe what’s in a photo and speak it out.")

# =========================
# INPUT SECTION
# =========================
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
            type=["png", "jpg", "jpeg", "webp"]
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
        format_func=lambda x: x[0],
    )[1]

    add_ocr = st.toggle(
        "Include text in the scene (OCR)",
        value=False
    )

st.divider()

# =========================
# IMAGE PREVIEW
# =========================
if img is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

preview = ImageOps.contain(img, (768, 768))
st.image(preview, caption="Input image", width="stretch")

# =========================
# INFERENCE (STREAMLIT → FASTAPI)
# =========================
if st.button("🔊 Describe & Speak", type="primary"):
    with st.spinner("Sending image to backend..."):
        try:
            start_time = time.time()

            # Convert image → bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            files = {
                "file": ("image.jpg", img_buffer, "image/jpeg")
            }

            data = {
                "lang": lang,
                "include_ocr": str(add_ocr).lower()
            }

            # ---- Call FastAPI (audio endpoint) ----
            response = requests.post(
                f"{API_BASE_URL}/predict/audio",
                files=files,
                data=data,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            caption = result["caption"]
            audio_bytes = io.BytesIO(
                bytes.fromhex(result["audio_hex"])
            )

            elapsed = time.time() - start_time

        except requests.exceptions.RequestException as e:
            st.error("❌ Backend connection failed.")
            st.code(str(e))
            st.stop()

    st.success("Done!")
    st.write("**Description:**", caption)
    st.audio(audio_bytes, format="audio/mp3")
    st.caption(f"Processed in ~{elapsed:.1f}s")

# =========================
# NOTES
# =========================
with st.expander("Design notes & future improvements"):
    st.markdown(
        "- Streamlit UI decoupled from ML backend using REST APIs\n"
        "- Backend deployed as Dockerized FastAPI service on AWS EC2\n"
        "- OCR and language selection handled via request parameters\n"
        "- Future: async requests, streaming audio, auth, HTTPS"
    )
