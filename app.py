import streamlit as st
import replicate
import requests
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Nike Video Generator", page_icon="🏃", layout="wide")
st.title("🏃 Nike Commercial Video Generator")
st.subheader("Wan 2.2 I2V-fast via Replicate API • No GPU needed!")
st.caption("Hybrid Image + Text → 720p promotional video | Commercial use allowed")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Model: wan-video/wan-2.2-i2v-fast (Wan2.2 5B hybrid family)\n"
            "Generation time: ~1-5 minutes\n"
            "Cost: ~$0.10–0.30 per video (your Replicate credits)")

# ================== MAIN ==================
uploaded_file = st.file_uploader("Upload Nike starting image (product/athlete/logo)", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Nike reference image", use_column_width=True)

prompt = st.text_area(
    "Marketing Prompt",
    value="Dynamic Nike athlete sprinting through a futuristic neon city at golden hour, energetic swoosh branding, cinematic sports commercial style, high energy motion, smooth camera pan, professional advertising look",
    height=150
)

if st.button("🚀 Generate Nike Promo Video", type="primary"):
    if not st.secrets.get("REPLICATE_API_TOKEN"):
        st.error("❌ Please add REPLICATE_API_TOKEN to .streamlit/secrets.toml")
        st.stop()

    replicate.api_key = st.secrets["REPLICATE_API_TOKEN"]

    with st.spinner("Generating on Replicate cloud... (1–5 minutes)"):
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        # Call the exact Wan 2.2 model
        output = replicate.run(
            "wan-video/wan-2.2-i2v-fast",
            input={
                "image": open(image_path, "rb"),   # Replicate auto-uploads the file
                "prompt": prompt,
                # You can add more params here if you want (check model page):
                # "num_frames": 81,
                # "fps": 24,
                # "resolution": "720p",
            }
        )

        # Cleanup temp file
        os.unlink(image_path)

        # Replicate usually returns the video URL directly (or list[0])
        video_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else output.get("video")

    st.success("✅ Nike commercial video ready!")
    st.video(video_url)

    # Download button
    video_data = requests.get(video_url).content
    st.download_button(
        label="📥 Download MP4 for your project",
        data=video_data,
        file_name="nike_commercial_720p.mp4",
        mime="video/mp4"
    )
