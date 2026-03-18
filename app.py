import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import os

# -----------------------------
# API KEY
# -----------------------------
HF_API_KEY = None
if "HF_API_KEY" in st.secrets:
    HF_API_KEY = st.secrets["HF_API_KEY"]
else:
    HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY is None:
    st.error("❌ Missing Hugging Face API Key")
    st.stop()

# -----------------------------
# HF API CALL
# -----------------------------
def query_hf(model, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# -----------------------------
# IMAGE EXTRACTION
# -----------------------------
def get_product_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"]

        img = soup.find("img")
        if img and img.get("src"):
            return img["src"]

        return None
    except:
        return None

def load_image(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return Image.open(BytesIO(res.content))
    except:
        return None

# -----------------------------
# IMAGE → DESCRIPTION (BLIP-2)
# -----------------------------
def generate_caption(image_url):
    payload = {"inputs": image_url}
    res = query_hf("Salesforce/blip-image-captioning-base", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "white Nike running shoes with modern design"

# -----------------------------
# CINEMATIC STORY GENERATION
# -----------------------------
def generate_cinematic_prompt(description, name, age, gender, nationality, city):

    # gender wording
    gender_word = "man" if gender == "Male" else "woman"

    prompt = f"""
You are a world-class Nike commercial director and cinematic storyteller.

Create a HIGH-END cinematic promotional scene description.

Product:
{description}

User:
- Name: {name}
- Age: {age}
- Gender: {gender_word}
- Nationality: {nationality}
- Location: {city}

STRICT REQUIREMENTS:
- Output ONLY one cinematic paragraph (no bullet points)
- Make it visually rich and emotionally powerful
- Include camera movements, lighting, atmosphere
- Mention the product in a detailed and stylish way
- Keep it similar to a Nike commercial
- Include motion, energy, and confidence
- No explanations

Style reference:
"Cinematic motivational running ad at dawn in Tokyo. A confident 28-year-old athlete runs through neon-lit streets..."

Now generate:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.9,
            "top_p": 0.95
        }
    }

    res = query_hf("mistralai/Mistral-7B-Instruct-v0.2", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "Cinematic running scene with Nike shoes, sunrise lighting, powerful motion."

# -----------------------------
# UI
# -----------------------------
st.title("🎬 AI Cinematic Nike Ad Generator")

product_url = st.text_input("Nike Product URL (optional)")
uploaded_file = st.file_uploader("OR Upload Product Image")

name = st.text_input("Name", "David")
age = st.slider("Age", 18, 50, 28)
gender = st.selectbox("Gender", ["Male", "Female"])
nationality = st.text_input("Nationality", "Chinese")
city = st.text_input("City", "Shanghai")

if st.button("Generate Cinematic Ad Prompt"):

    # -----------------------------
    # GET IMAGE
    # -----------------------------
    img = None

    if product_url:
        img_url = get_product_image(product_url)
        if img_url:
            img = load_image(img_url)

    if uploaded_file:
        img = Image.open(uploaded_file)

    if img is None:
        st.warning("⚠️ Using fallback product description")

    else:
        st.image(img)

    # -----------------------------
    # DESCRIPTION
    # -----------------------------
    st.write("🧠 Understanding product...")
    description = generate_caption(product_url if product_url else "Nike shoes")
    st.write("**Detected product:**", description)

    # -----------------------------
    # GENERATE CINEMATIC PROMPT
    # -----------------------------
    st.write("🎥 Generating cinematic storyline...")

    cinematic_prompt = generate_cinematic_prompt(
        description, name, age, gender, nationality, city
    )

    st.success("✨ Cinematic Prompt Generated")

    st.text_area("🎬 Final Cinematic Prompt", cinematic_prompt, height=250)
