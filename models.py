from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast = True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model
