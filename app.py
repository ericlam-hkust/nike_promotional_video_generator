import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import time

# fal-client reads FAL_KEY from env automatically
if "FAL_KEY" in st.secrets:
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
else:
    st.error("❌ FAL_KEY not found in .streamlit/secrets.toml or Streamlit secrets")
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
    if not uploaded_file:
        st.error("Please upload a starting Nike image first.")
        st.stop()

    with st.spinner("Uploading image + generating on fal.ai... (30–150 seconds)"):
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        try:
            # Step 1: Upload image to fal storage → get public URL
            image_url = fal.storage.upload(image_path)  # Returns https://... URL
            st.info(f"Image uploaded to: {image_url}")

            # Step 2: Run the model with image_url
            result = fal.subscribe(
                "fal-ai/wan/v2.2-5b/image-to-video",
                arguments={
                    "image_url": image_url,
                    "prompt": prompt,
                    # Optional params (add as needed from model API docs):
                    # "negative_prompt": "blurry, lowres, artifacts, deformed",
                    # "num_inference_steps": 50,
                    # "guidance_scale": 7.5,
                },
                timeout=300,  # 5 minutes
            )

            # Cleanup local temp file
            os.unlink(image_path)

            # Extract video URL (fal usually returns dict with 'video' → {'url': ...})
            if isinstance(result, dict):
                video_data = result.get("video", {})
                video_url = video_data.get("url") if isinstance(video_data, dict) else None
            elif isinstance(result, list) and result:
                video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]
            else:
                video_url = None

            if not video_url:
                st.error("No valid video URL in response. Check fal.ai dashboard/logs.")
                st.json(result)  # Debug output
                st.stop()

        except Exception as e:
            st.error(f"fal.ai error: {str(e)}")
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
        st.warning(f"Video ready in player above — right-click to save if download fails. ({str(e)})")
