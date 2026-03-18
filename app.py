import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import os

# -----------------------------
# API KEY SETUP
# -----------------------------
HF_API_KEY = None
if "HF_API_KEY" in st.secrets:
    HF_API_KEY = st.secrets["HF_API_KEY"]
else:
    HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY is None:
    st.error("❌ Hugging Face API key not found.")
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
# IMAGE EXTRACTION FROM URL
# -----------------------------
def extract_image_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Try Open Graph image first
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"]

        # Fallback: first image tag
        img = soup.find("img")
        if img and img.get("src"):
            return img["src"]

        return None
    except:
        return None

# -----------------------------
# SAFE IMAGE LOADING
# -----------------------------
def load_image_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return Image.open(BytesIO(res.content))
    except:
        return None

# -----------------------------
# IMAGE → DESCRIPTION (BLIP)
# -----------------------------
def generate_caption(image_url):
    payload = {"inputs": image_url}
    res = query_hf("Salesforce/blip-image-captioning-base", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "white Nike running shoes with modern design"

# -----------------------------
# CINEMATIC PROMPT GENERATION
# -----------------------------
def generate_cinematic_storyboard(description, name, age, gender, nationality, city):

    gender_word = "man" if gender == "Male" else "woman"

    prompt = f"""
You are a world-class film director creating a Nike commercial.

Create a cinematic storyboard with a timeline.

Product:
{description}

Character:
{name}, a {age}-year-old {nationality} {gender_word} in {city}

STRICT OUTPUT FORMAT:

[0-3s]
Shot:
Camera:
Visual:
Emotion:

[3-6s]
Shot:
Camera:
Visual:
Emotion:

[6-10s]
Shot:
Camera:
Visual:
Emotion:

[10-15s]
Shot:
Camera:
Visual:
Emotion:

Final Slogan:

STYLE REQUIREMENTS:
- Nike commercial style
- Cinematic, dramatic lighting
- Include motion (running, energy, impact)
- Include product details (color, texture, glow)
- Use professional film language (tracking shot, slow motion, close-up, wide shot)
- No explanations outside format

Generate:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.85,
            "top_p": 0.95
        }
    }

    # 🔥 BEST MODEL (if available)
    model = "meta-llama/Meta-Llama-3-70B-Instruct"

    # fallback if rate limited:
    # model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    res = query_hf(model, payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "Storyboard generation failed."

# -----------------------------
# UI
# -----------------------------
st.title("🎬 AI Cinematic Nike Ad Generator")

st.markdown("### 📥 Provide Product Input")

input_option = st.radio(
    "Choose input method:",
    ["🔗 Use Product URL", "📤 Upload Image"]
)

product_url = None
uploaded_file = None
image = None

# -----------------------------
# OPTION 1: URL
# -----------------------------
if input_option == "🔗 Use Product URL":
    product_url = st.text_input("Enter Product URL")

    if product_url:
        st.write("🔍 Extracting image from URL...")
        img_url = extract_image_from_url(product_url)

        if img_url:
            image = load_image_from_url(img_url)

            if image:
                st.image(image, caption="Extracted Product Image")
            else:
                st.warning("⚠️ Could not load extracted image.")
        else:
            st.warning("⚠️ Could not find image in URL.")

# -----------------------------
# OPTION 2: UPLOAD
# -----------------------------
else:
    uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

# -----------------------------
# USER PROFILE
# -----------------------------
st.markdown("### 👤 User Profile")

name = st.text_input("Name", "David")
age = st.slider("Age", 18, 50, 28)
gender = st.selectbox("Gender", ["Male", "Female"])
nationality = st.text_input("Nationality", "Chinese")
city = st.text_input("City", "Shanghai")

# -----------------------------
# GENERATE
# -----------------------------
if st.button("🚀 Generate Cinematic Ad"):

    if image is None:
        st.error("❌ Please provide a product image (URL or upload).")
        st.stop()

    st.write("🧠 Understanding product...")

    # Use URL if available, otherwise fallback text
    image_input = product_url if product_url else "Nike product image"

    description = generate_caption(image_input)

    st.write("**Detected product:**", description)

    st.write("🎬 Generating cinematic storyline...")

    cinematic_prompt = generate_cinematic_prompt(
        description, name, age, gender, nationality, city
    )

    st.success("✨ Done!")

    st.text_area("🎥 Cinematic Ad Prompt", cinematic_prompt, height=250)
