import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import tempfile

HF_API_KEY = st.secrets["HF_API_KEY"]

# -----------------------------
# Hugging Face API helper
# -----------------------------
def query_hf(model, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# -----------------------------
# Extract product image (simple)
# -----------------------------
def get_product_image(url):
    try:
        html = requests.get(url).text
        # naive extraction
        start = html.find('property="og:image" content="') + 35
        end = html.find('"', start)
        img_url = html[start:end]
        return img_url
    except:
        return None

# -----------------------------
# Image caption (BLIP)
# -----------------------------
def generate_caption(image_url):
    payload = {"inputs": image_url}
    res = query_hf("Salesforce/blip-image-captioning-base", payload)
    return res.json()[0]["generated_text"]

# -----------------------------
# Generate marketing script (LLM)
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
    return res.json()[0]["generated_text"]

# -----------------------------
# Generate images (Stable Diffusion)
# -----------------------------
def generate_image(prompt):
    payload = {"inputs": prompt}
    res = query_hf("stabilityai/stable-diffusion-2", payload)

    if res.status_code == 200:
        return Image.open(BytesIO(res.content))
    else:
        return None

# -----------------------------
# Create video from images
# -----------------------------
def create_video(images):
    clips = []
    for img in images:
        temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_img.name)

        clip = ImageClip(temp_img.name).set_duration(3)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video.write_videofile(temp_video.name, fps=24)

    return temp_video.name

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏃 AI Nike Marketing Video Generator")

product_url = st.text_input("Nike Product URL")
name = st.text_input("User Name")
age = st.slider("Age", 10, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
nationality = st.text_input("Nationality")

if st.button("Generate Video"):

    user_profile = f"{age} year old {gender} from {nationality} named {name}"

    st.write("### Step 1: Extracting product image...")
    img_url = get_product_image(product_url)

    if not img_url:
        st.error("Could not extract product image")
        st.stop()

    st.image(img_url)

    st.write("### Step 2: Generating product description...")
    description = generate_caption(img_url)
    st.write(description)

    st.write("### Step 3: Generating marketing script...")
    script = generate_script(description, user_profile)
    st.write(script)

    st.write("### Step 4: Generating visuals...")
    prompts = [
        f"{description}, cinematic sports ad",
        f"{user_profile} running wearing nike shoes, dynamic lighting",
        "close-up nike shoes, dramatic lighting"
    ]

    images = []
    for p in prompts:
        img = generate_image(p)
        if img:
            st.image(img)
            images.append(img)

    st.write("### Step 5: Creating video...")
    video_path = create_video(images)

    st.video(video_path)

    st.success("✅ Video generated!")
