import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model

model = load_model()

st.title("🖼️ Text-to-Image Generator")

prompt = st.text_input("Enter a prompt:", "A fantasy landscape with mountains and a river")
generate = st.button("Generate Image")

if generate and prompt:
    with st.spinner("Generating image..."):
        image = model(prompt).images[0]
    st.image(image, caption="Generated Image", use_column_width=True)
