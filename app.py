import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import time

st.set_page_config(page_title="Nike Video Generator (fal.ai)", page_icon="🏃", layout="wide")
st.title("🏃 Nike Commercial Video Generator")
st.subheader("Wan 2.2 5B Image-to-Video via fal.ai • No GPU needed!")
st.caption("Upload Nike image + prompt → 720p ~5s promo video | Commercial use OK")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Model: fal-ai/wan/v2.2-5b/image-to-video\n"
            "Generation: ~30–120 seconds\n"
            "Cost: ~$0.05–0.30 per video (check fal dashboard)")

# ================== MAIN UI ==================
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
    if not st.secrets.get("FAL_KEY"):
        st.error("❌ Add FAL_KEY to .streamlit/secrets.toml")
        st.stop()

    fal.config(credentials=st.secrets["FAL_KEY"])

    with st.spinner("Generating on fal.ai cloud... (30–120 seconds)"):
        # Save image temporarily (fal accepts file-like or URL, but easiest with temp file)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        # Run the model (subscribe pattern — waits for completion)
        try:
            result = fal.subscribe(
                "fal-ai/wan/v2.2-5b/image-to-video",
                arguments={
                    "image_url": None,  # We'll upload the file instead
                    "prompt": prompt,
                    # Optional params (uncomment/adjust as needed):
                    # "num_frames": 121,
                    # "fps": 24,
                    # "negative_prompt": "blurry, low quality, deformed, static",
                },
                files={
                    "image": open(image_path, "rb"),  # Uploads your Nike image
                },
                # Optional: timeout longer if needed
                timeout=300,
            )

            # Cleanup temp file
            os.unlink(image_path)

            # fal result usually has 'video' key with URL
            video_url = result.get("video", {}).get("url") if isinstance(result, dict) else result[0].get("url") if isinstance(result, list) else None

            if not video_url:
                st.error("No video URL in response. Check fal dashboard/logs.")
                st.json(result)  # Debug: show full response
                st.stop()

        except Exception as e:
            st.error(f"fal.ai error: {str(e)}")
            st.stop()

    st.success("✅ Nike commercial video generated!")
    st.video(video_url)

    # Download button
    try:
        video_data = requests.get(video_url, timeout=30).content
        st.download_button(
            label="📥 Download MP4 for your project",
            data=video_data,
            file_name="nike_commercial_720p.mp4",
            mime="video/mp4"
        )
    except:
        st.warning("Video ready — right-click the player above to save, or check fal dashboard.")
