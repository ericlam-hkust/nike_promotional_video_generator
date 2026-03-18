import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import tempfile
import os
import imageio_ffmpeg
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Fix ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

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
# HF API helper
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

def safe_load_image(url):
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
        return "A high-performance Nike product"

# -----------------------------
# STORYLINE GENERATION (LLM)
# -----------------------------
def generate_storyline(description, user_profile):
    prompt = f"""
    You are a Nike marketing expert.

    Product: {description}
    Audience: {user_profile}

    Create a cinematic promotional storyline.

    Output format:
    Scene 1:
    Scene 2:
    Scene 3:
    Final slogan:
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }

    res = query_hf("mistralai/Mistral-7B-Instruct-v0.2", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "Scene 1: Athlete starts running\nScene 2: Close-up shoes\nScene 3: Victory moment\nFinal slogan: Just Do It"

# -----------------------------
# PARSE STORY INTO SCENES
# -----------------------------
def extract_scenes(story):
    lines = story.split("\n")
    scenes = []

    for line in lines:
        if "Scene" in line:
            scenes.append(line.split(":")[-1].strip())

    return scenes[:3]  # limit to 3 scenes

# -----------------------------
# IMAGE GENERATION (SD)
# -----------------------------
def generate_image(prompt):
    payload = {"inputs": prompt}
    res = query_hf("stabilityai/stable-diffusion-2", payload)

    if res.status_code == 200:
        return Image.open(BytesIO(res.content))
    return None

# -----------------------------
# VIDEO CREATION
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
st.title("🎬 AI Personalized Marketing Video Generator")

product_url = st.text_input("Nike Product URL (optional)")
uploaded_file = st.file_uploader("OR upload product image")

name = st.text_input("Name")
age = st.slider("Age", 10, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
nationality = st.text_input("Nationality")

if st.button("Generate Marketing Video"):

    user_profile = f"{age} year old {gender} from {nationality} named {name}"

    # -----------------------------
    # GET IMAGE
    # -----------------------------
    img = None

    if product_url:
        img_url = get_product_image(product_url)
        if img_url:
            img = safe_load_image(img_url)

    if uploaded_file:
        img = Image.open(uploaded_file)

    if img is None:
        st.warning("⚠️ Using fallback AI image")
        img = generate_image("Nike running shoes product photo")

    st.image(img)

    # -----------------------------
    # DESCRIPTION
    # -----------------------------
    st.write("🧠 Understanding product...")
    description = generate_caption(product_url if product_url else "Nike product")
    st.write(description)

    # -----------------------------
    # STORYLINE
    # -----------------------------
    st.write("✍️ Generating storyline...")
    story = generate_storyline(description, user_profile)
    st.text(story)

    # -----------------------------
    # SCENES
    # -----------------------------
    scenes = extract_scenes(story)

    st.write("🎨 Generating scenes...")

    generated_images = []

    for scene in scenes:
        prompt = f"{scene}, cinematic nike advertisement, dramatic lighting"
        img_gen = generate_image(prompt)

        if img_gen:
            st.image(img_gen)
            generated_images.append(img_gen)

    if not generated_images:
        generated_images = [img]

    # -----------------------------
    # VIDEO
    # -----------------------------
    st.write("🎬 Creating video...")
    video_path = create_video(generated_images)

    st.video(video_path)

    st.success("✅ Done!")
