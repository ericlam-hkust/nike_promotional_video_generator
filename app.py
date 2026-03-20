import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import base64
import io

# Set FAL_KEY from secrets (fal-client reads from env)
if "FAL_KEY" in st.secrets:
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
else:
    st.error("❌ FAL_KEY not found in .streamlit/secrets.toml or Streamlit app secrets.")
    st.stop()

st.set_page_config(page_title="Nike Video Generator (Kling 3.0 Pro)", page_icon="🏃", layout="wide")
st.title("🏃 Nike Commercial Video Generator – Kling 3.0 Pro")
st.subheader("High-quality 1080p Image-to-Video via fal.ai • No local GPU!")
st.caption("Kling 3.0 Pro: Cinematic motion, fluid dynamics, strong prompt adherence | Up to 15s clips | Commercial use OK")

# ================== SIDEBAR SETTINGS ==================
with st.sidebar:
    st.header("⚙️ Generation Settings")
    st.info("Model: fal-ai/kling-video/v3/pro/image-to-video\n"
            "Output: Up to 1080p native in Pro mode\n"
            "Generation: ~60–300 seconds | Cost: ~$0.10–0.30 per clip")
    
    duration = st.slider("Duration (seconds)", 3, 15, 10, help="Kling 3.0 Pro supports 3–15s; higher = more cost")
    aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0, help="16:9 for cinematic promo")
    cfg_scale = st.slider("Guidance Scale (CFG)", 0.1, 10.0, 3.0, 0.5, help="Higher = stricter prompt following; 2.5–5.0 often best for motion")
    negative_prompt = st.text_area("Negative Prompt (avoid these)", 
                                   value="blurry, low quality, artifacts, deformed, static, text, watermark, ugly, distorted, overexposed",
                                   height=100)

# ================== MAIN UI ==================
uploaded_file = st.file_uploader("Upload starting Nike image (product, athlete, logo, etc.)", 
                                 type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Nike reference image", use_column_width=True)

prompt = st.text_area(
    "Marketing / Motion Prompt",
    value="Dynamic Nike athlete powerfully sprinting forward through a futuristic neon-lit city at golden hour sunset, energetic visible swoosh branding on clothing and billboards, cinematic sports commercial style, high-energy fluid motion, smooth dramatic tracking camera pan and follow shot, professional advertising look, intense dramatic lighting, high detail",
    height=150
)

if st.button("🚀 Generate Nike Promo Video (Kling 3.0 Pro)", type="primary"):
    if not uploaded_file:
        st.error("Please upload a starting Nike image first.")
        st.stop()

    with st.spinner("Encoding image + generating high-quality video on fal.ai... (1–5 minutes)"):
        try:
            # Convert image to base64 data URL (JPEG for compatibility/size)
            buffered = io.BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{img_base64}"

            # Run Kling 3.0 Pro I2V
            result = fal.subscribe(
                "fal-ai/kling-video/v3/pro/image-to-video",
                arguments={
                    "prompt": prompt,
                    "start_image_url": image_data_url,  # Kling uses start_image_url for I2V
                    "duration": str(duration),          # Must be string per schema
                    "aspect_ratio": aspect_ratio,
                    "negative_prompt": negative_prompt,
                    "cfg_scale": cfg_scale,
                    # Optional extras (uncomment if needed):
                    # "enable_audio": False,  # Kling Pro can generate native audio if True
                    # "mode": "professional",  # Some variants support modes
                },
                timeout=600,  # 10 min max for longer/higher-quality gens
            )

            # Extract video URL (Kling returns dict with 'video' → {'url': ...})
            video_url = None
            if isinstance(result, dict):
                video_data = result.get("video", {})
                video_url = video_data.get("url") if isinstance(video_data, dict) else result.get("video_url")
            elif isinstance(result, list) and result:
                video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]

            if not video_url:
                st.error("No valid video URL returned. Check fal.ai dashboard/logs.")
                st.json(result)  # Debug: full response
                st.stop()

        except Exception as e:
            st.error(f"fal.ai / Kling generation failed: {str(e)}")
            st.stop()

    st.success("✅ High-quality Nike commercial video generated with Kling 3.0 Pro!")
    st.video(video_url)

    # Download button
    try:
        video_response = requests.get(video_url, timeout=60)
        video_response.raise_for_status()
        st.download_button(
            label="📥 Download 1080p MP4 for your project",
            data=video_response.content,
            file_name=f"nike_kling_promo_{duration}s.mp4",
            mime="video/mp4"
        )
    except Exception as e:
        st.warning("Video playable above — right-click player to save if download fails.")
