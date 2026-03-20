import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import base64
import io

# fal-client reads FAL_KEY from environment automatically
if "FAL_KEY" in st.secrets:
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
else:
    st.error("❌ FAL_KEY not found in .streamlit/secrets.toml or Streamlit app secrets.")
    st.stop()

st.set_page_config(page_title="Nike Video Generator (fal.ai)", page_icon="🏃", layout="wide")
st.title("🏃 Nike Commercial Video Generator")
st.subheader("kling-video/v3/pro/image-to-video via fal.ai • Cloud-based, no local GPU!")
st.caption("Upload Nike image + prompt → ~15s 1080p promo video | Commercial use allowed")

# ================== SIDEBAR SETTINGS ==================
with st.sidebar:
    st.header("⚙️ Generation Settings")
    st.info("Model: fal-ai/kling-video/v3/pro/image-to-video")
    num_frames = st.slider("Number of frames", 17, 161, 161, help="~81 frames ≈ 3-4 seconds at 24 fps")
    fps = st.slider("Frames per second", 4, 60, 10)
    resolution = st.selectbox("Resolution", ["1080p", "720p", "580p"], index=0)
    guidance_scale = st.slider("Guidance scale", 1.0, 10.0, 3.5, 0.5)
    negative_prompt = st.text_area("Negative prompt (optional)", 
                                   value="blurry, low quality, artifacts, deformed, static, text, watermark, ugly",
                                   height=100)
    st.caption("Generation time: ~30–150 seconds | Cost: low (~$0.05–0.30)")

# ================== MAIN UI ==================
uploaded_file = st.file_uploader("Upload starting Nike image (product, athlete, logo, etc.)", 
                                 type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Nike reference image", use_column_width=True)

prompt = st.text_area(
    "Marketing / Motion Prompt",
    value="Dynamic Nike athlete sprinting powerfully through a futuristic neon city at golden hour, energetic swoosh branding visible, cinematic sports commercial style, high energy motion, smooth dramatic camera pan and follow, professional advertising look, intense lighting",
    height=150
)

if st.button("🚀 Generate Nike Promo Video", type="primary"):
    if not uploaded_file:
        st.error("Please upload a starting Nike image first.")
        st.stop()

    with st.spinner("Encoding image + generating on fal.ai cloud... (30–150 seconds)"):
        try:
            # Convert image to base64 data URL
            buffered = io.BytesIO()
            # Convert to JPEG for smaller size / better compatibility
            image.convert("RGB").save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{img_base64}"

            # Run the model
            result = fal.subscribe(
                "fal-ai/wan/v2.2-5b/image-to-video",
                arguments={
                    "image_url": image_data_url,
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "frames_per_second": fps,
                    "resolution": resolution.lower(),
                    "negative_prompt": negative_prompt,
                    "guidance_scale": guidance_scale,
                    # Add more if desired (see model API docs):
                    # "seed": 42,
                    # "enable_safety_checker": True,
                    # "video_quality": "high",
                }
            )

            # Extract video URL from result
            video_url = None
            if isinstance(result, dict):
                video_data = result.get("video", {})
                video_url = video_data.get("url") if isinstance(video_data, dict) else result.get("video_url")
            elif isinstance(result, list) and result:
                video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]

            if not video_url:
                st.error("No video URL found in response.")
                st.json(result)  # Show full output for debugging
                st.stop()

        except Exception as e:
            st.error(f"fal.ai generation failed: {str(e)}")
            st.stop()

    st.success("✅ Nike commercial video ready!")
    st.video(video_url)

    # Download button
    try:
        video_response = requests.get(video_url, timeout=30)
        video_response.raise_for_status()
        st.download_button(
            label="📥 Download MP4 for your master's project",
            data=video_response.content,
            file_name="nike_promo_video_720p.mp4",
            mime="video/mp4"
        )
    except Exception as e:
        st.warning("Video is playable above — right-click the player to save if download button fails.")
