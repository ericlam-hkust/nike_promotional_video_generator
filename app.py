import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import tempfile
import os
import imageio_ffmpeg

# Fix ffmpeg for Streamlit Cloud
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# -----------------------------
# API Key Handling (robust)
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
# Hugging Face API helper
# -----------------------------
def query_hf(model, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# -----------------------------
# Extract product image safely
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

# -----------------------------
# Safe image loader (NO CRASH)
# -----------------------------
def safe_load_image(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        img = Image.open(BytesIO(res.content))
        return img
    except:
        return None

# -----------------------------
# Image caption (BLIP)
# -----------------------------
def generate_caption(image_url):
    payload = {"inputs": image_url}
    res = query_hf("Salesforce/blip-image-captioning-base", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "A stylish Nike product"

# -----------------------------
# Generate marketing script
# -----------------------------
def generate_script(description, user_profile):
    prompt = f"""
    Create a 15-second Nike-style ad.

    Product: {description}
    Audience: {user_profile}

    Output:
    - Hook
    - Scene 1
    - Scene 2
    - Call to action
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }

    res = query_hf("mistralai/Mistral-7B-Instruct-v0.2", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "Just do it. Experience the next level of performance."

# -----------------------------
# Generate image (Stable Diffusion)
# -----------------------------
def generate_image(prompt):
    payload = {"inputs": prompt}
    res = query_hf("stabilityai/stable-diffusion-2", payload)

    if res.status_code == 200:
        return Image.open(BytesIO(res.content))
    return None

# -----------------------------
# Create video (stable version)
# -----------------------------
def create_video(images):
    temp_files = []

    for img in images:
        temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp.name)
        temp_files.append(temp.name)

    clip = ImageSequenceClip(temp_files, fps=1)

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    clip.write_videofile(temp_video.name)

    return temp_video.name

# -----------------------------
# UI
# -----------------------------
st.title("🏃 AI Marketing Video Generator")

product_url = st.text_input("Nike Product URL (optional)")
uploaded_file = st.file_uploader("OR upload product image")

name = st.text_input("User Name")
age = st.slider("Age", 10, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
nationality = st.text_input("Nationality")

if st.button("Generate Video"):

    user_profile = f"{age} year old {gender} from {nationality} named {name}"

    img = None

    # -----------------------------
    # Step 1: Get image
    # -----------------------------
    if product_url:
        st.write("🔍 Extracting product image...")
        img_url = get_product_image(product_url)
        st.write("DEBUG URL:", img_url)

        if img_url:
            img = safe_load_image(img_url)

    if uploaded_file:
        img = Image.open(uploaded_file)

    if img is None:
        st.warning("⚠️ Could not get product image. Using AI-generated fallback.")
        img = generate_image("Nike running shoes, product photography")

    st.image(img)

    # -----------------------------
    # Step 2: Caption
    # -----------------------------
    st.write("🧠 Generating description...")
    description = generate_caption(product_url if product_url else "Nike product")
    st.write(description)

    # -----------------------------
    # Step 3: Script
    # -----------------------------
    st.write("✍️ Generating script...")
    script = generate_script(description, user_profile)
    st.write(script)

    # -----------------------------
    # Step 4: Generate visuals
    # -----------------------------
    st.write("🎨 Generating visuals...")

    prompts = [
        f"{description}, cinematic sports ad",
        f"{user_profile} running wearing nike shoes, dynamic lighting",
        "close-up nike shoes, dramatic lighting"
    ]

    images = []

    for p in prompts:
        img_gen = generate_image(p)
        if img_gen:
            st.image(img_gen)
            images.append(img_gen)

    # fallback if no images
    if not images:
        images = [img]

    # -----------------------------
    # Step 5: Video
    # -----------------------------
    st.write("🎬 Creating video...")
    video_path = create_video(images)

    st.video(video_path)

    st.success("✅ Video generated successfully!")
