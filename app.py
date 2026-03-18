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
def generate_cinematic_prompt(description, name, age, gender, nationality, city):

    gender_word = "man" if gender == "Male" else "woman"

    prompt = f"""
You are a world-class Nike commercial director.

Create a cinematic promotional ad scene.

Product:
{description}

User:
{name}, {age}-year-old {nationality} {gender_word} in {city}

Requirements:
- At least two hundred words
- Shot-by-Shot Script with Timing
- Cinematic, emotional, high-end
- Include camera motion, lighting, energy
- Describe the product in detail
- Nike-style storytelling

Example style:
"0:00–0:05 – Opening hook
Visuals: Fade in from black to aerial drone shot over Shanghai at pre-dawn. Huangpu River glows faintly with city lights. Cut to close-up of alarm clock showing 5:45. A hand (Asian male, mid-30s) turns it off. Camera pans to running shoes (your Nike ReactX ZoomX: white upper, bold volt-green midsole/accents, black Swoosh, orange heel flash) placed by the bed—neon details subtly glowing in low light.
Voiceover (deep, motivational male voice, slight echo): "In the early dawn of Shanghai, 35-year-old Wei has made the 5:45 alarm a ritual… not a chore."
Sound: Soft ambient city hum + gentle rising synth pad.
0:05–0:12 – Lacing up & energy build
Visuals: Slow-motion close-ups: Hands tying laces (ReactX foam compressing softly). Pull back to full body—Wei in simple running gear standing by window, city skyline behind. Quick cuts: Volt-green midsole details lighting up as he steps. Zoom on black Swoosh + orange heel flash catching first sunlight.
Voiceover: "It's the quiet thrill of slipping out before the city wakes… lacing up these Nike ReactX ZoomX shoes… and hitting the pavement."
Sound: Heartbeat pulse + subtle energy-return "whoosh" effect on steps.
0:12–0:22 – The run
Visuals: Dynamic tracking shots along the Bund riverside path. Wei running smoothly—fluid strides, confident form. Intercut:

Slow-mo footstrikes (ZoomX foam compressing then exploding forward with energy trails in volt green).
Close-ups of the vibrant midsole flashing in sunrise light.
Wide shots: Runners glancing over, impressed by the shoe's bold design.
POV: Wei's view—path ahead, city awakening.
Voiceover: "At 35, you're not chasing PRs every run anymore. You've learned smarter running: protecting your body while still pushing it. ReactX hugs your feet… ZoomX turns every impact into forward propulsion. 'You've got more in you today.'"
Sound: Building upbeat electronic track (think Nike "Just Do It" style—driving bass, motivational synths). Footstrike impacts with satisfying bounce SFX.

0:22–0:32 – Climax acceleration
Visuals: Wei opens up stride for a strong finish—faster cuts, dynamic angles. Slow-mo burst: Volt-green midsole glowing intensely, orange heel flashing like a spark. Cut to heroic side profile running toward rising sun. Quick montage: Sweat beads, determined expression, city blurring past.
Voiceover: "Work hard, yes—but carve out time for health. Stress builds up? Run it out. Life speeds up? Keep your own rhythm. The second half of life is for running smarter… stronger… brighter."
Sound: Music peaks with powerful drop + crowd-like whoosh energy.
0:32–0:40 – Close & call-to-action
Visuals: Wei slows to a powerful stop, breathing hard but smiling. Close-up on shoes—volt accents popping. Product lock-up: Full shoe reveal (your image style) with text overlay: "Nike ReactX ZoomX" + tagline fade in. Final frame: Nike swoosh + "Just Do It."
Voiceover: "These aren't just shoes. They're your 35-year-old declaration: Still fast. Still strong. Still you. In China, every step counts."
On-screen text (final 5s): "Nike ReactX ZoomX – Available now. Just Do It."
Sound: Music resolves uplifting, ends on clean hit."

Generate:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.9
        }
    }

    res = query_hf("mistralai/Mistral-7B-Instruct-v0.2", payload)

    try:
        return res.json()[0]["generated_text"]
    except:
        return "Cinematic Nike running scene with powerful motion."

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
