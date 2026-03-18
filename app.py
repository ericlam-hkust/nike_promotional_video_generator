import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import os
import time

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
# SAFE HF API CALL (NEW ROUTER)
# -----------------------------
def query_with_retry(model, payload, retries=5):
    API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(retries):
        try:
            res = requests.post(API_URL, headers=headers, json=payload, timeout=30)

            # Handle non-JSON response
            try:
                data = res.json()
            except:
                print("⚠️ Non-JSON response:", res.text[:200])
                time.sleep(3)
                continue

            # Handle HF errors
            if isinstance(data, dict):
                if "error" in data:
                    print("⚠️ HF error:", data["error"])
                    time.sleep(5)
                    continue

            # Success
            if isinstance(data, list):
                return data

        except requests.exceptions.RequestException as e:
            print("⚠️ Request failed:", str(e))
            time.sleep(3)

    return None

# -----------------------------
# IMAGE EXTRACTION FROM URL
# -----------------------------
def extract_image_from_url(url):
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

# -----------------------------
# LOAD IMAGE SAFELY
# -----------------------------
def load_image(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return Image.open(BytesIO(res.content))
    except:
        return None

# -----------------------------
# IMAGE → DESCRIPTION
# -----------------------------
def generate_caption(image_url):
    payload = {"inputs": image_url}
    data = query_with_retry("Salesforce/blip-image-captioning-base", payload)

    if data:
        try:
            return data[0]["generated_text"]
        except:
            return "Nike running shoes with modern sporty design"
    else:
        return "Nike running shoes with modern sporty design"

# -----------------------------
# STORYBOARD GENERATION
# -----------------------------
def generate_cinematic_storyboard(description, name, age, gender, nationality, city):

    gender_word = "man" if gender == "Male" else "woman"

    prompt = f"""
You are a professional film director creating a Nike commercial storyboard.

Product:
{description}

Character:
{name}, a {age}-year-old {nationality} {gender_word} in {city}

Create a cinematic timeline.

STRICT FORMAT:

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

Rules:
- Cinematic and detailed
- Include lighting, motion, product detail
- No explanation outside format
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.8
        }
    }

    model = "mistralai/Mistral-7B-Instruct-v0.2"

    data = query_with_retry(model, payload)

    if data:
        try:
            return data[0]["generated_text"]
        except:
            return "⚠️ Parsing error. Try again."
    else:
        return "⚠️ API failed or model unavailable. Please try again."

# -----------------------------
# UI
# -----------------------------
st.title("🎬 AI Cinematic Nike Ad Generator")

st.markdown("### 📥 Product Input")

input_option = st.radio(
    "Choose input method:",
    ["🔗 Use Product URL", "📤 Upload Image"]
)

product_url = None
image = None

# -----------------------------
# OPTION 1: URL
# -----------------------------
if input_option == "🔗 Use Product URL":
    product_url = st.text_input("Enter Product URL")

    if product_url:
        st.write("🔍 Extracting image...")
        img_url = extract_image_from_url(product_url)

        if img_url:
            image = load_image(img_url)

            if image:
                st.image(image, caption="Extracted Image")
            else:
                st.warning("⚠️ Failed to load image from URL.")
        else:
            st.warning("⚠️ No image found at URL.")

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
# GENERATE BUTTON
# -----------------------------
if st.button("🚀 Generate Cinematic Storyboard"):

    if image is None:
        st.error("❌ Please provide a product image.")
        st.stop()

    st.write("🧠 Understanding product...")

    image_input = product_url if product_url else "Nike running shoes"
    description = generate_caption(image_input)

    st.write("**Detected product:**", description)

    st.write("🎬 Generating storyboard...")

    storyboard = generate_cinematic_storyboard(
        description, name, age, gender, nationality, city
    )

    st.success("✨ Done!")

    st.text_area("🎥 Cinematic Storyboard", storyboard, height=350)
