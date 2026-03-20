import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import time

# Set the FAL_KEY as environment variable (fal-client reads this automatically)
if "FAL_KEY" in st.secrets:
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
else:
    st.error("❌ FAL_KEY not found in .streamlit/secrets.toml")
    st.stop()

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
    with st.spinner("Generating on fal.ai cloud... (30–120 seconds)"):
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        try:
            # Run the model (subscribe waits for completion)
            result = fal.subscribe(
                "fal-ai/wan/v2.2-5b/image-to-video",
                arguments={
                    "prompt": prompt,
                    # Optional: add more if needed (check https://fal.ai/models/fal-ai/wan/v2.2-5b/image-to-video/api)
                    # "negative_prompt": "blurry, low quality, artifacts",
                    # "num_frames": 121,
                    # "fps": 24,
                },
                files={
                    "image": open(image_path, "rb"),  # Uploads your Nike image
                },
                timeout=300,  # 5 min timeout
            )

            # Cleanup
            os.unlink(image_path)

            # Extract video URL from result (fal returns dict with 'video' key usually)
            if isinstance(result, dict):
                video_url = result.get("video", {}).get("url")
            elif isinstance(result, list) and result:
                video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]
            else:
                video_url = None

            if not video_url:
                st.error("No valid video URL returned. Check fal.ai dashboard for logs.")
                st.json(result)  # Show full response for debug
                st.stop()

        except Exception as e:
            st.error(f"fal.ai generation failed: {str(e)}")
            if os.path.exists(image_path):
                os.unlink(image_path)
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
    except Exception as e:
        st.warning(f"Video ready (player above) — right-click to save manually if download fails. ({str(e)})")
