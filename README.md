# Hear The Image (Image → Audio Scene Describer)  

An **AI-powered accessibility tool** that helps blind and low-vision users understand images.  
Upload or capture an image, and the app will **describe the scene in natural language** and **speak it out loud** in your chosen language.  

---

## 🚀 Features  
- 🖼️ **Scene Understanding** — Generates natural captions for any image using BLIP.  
- 🔊 **Speech Output** — Converts the caption into audio with Google Text-to-Speech.  
- 📝 **Optional OCR** — Reads printed text inside images (EasyOCR).  
- 🌍 **Multilingual Support** — English, Hindi, Telugu, Tamil, Kannada.  
- 📸 **Camera/Upload Input** — Works with uploads or live camera captures in Streamlit.  

---

## 🛠️ Tech Stack  
- [Streamlit](https://streamlit.io) — Interactive UI  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — BLIP captioning model  
- [gTTS](https://pypi.org/project/gTTS/) — Google Text-to-Speech  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) *(optional)* — Text recognition in images  

---

## 📦 Installation  

```bash
# Clone the repository
git clone https://github.com/sujithvaishnav/Hear_the_image.git
cd Hear_the_image

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

streamlit run app.py
