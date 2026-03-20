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
    cfg_scale = st.slider("Guidance Scale (CFG)", 0.1, 1.0, 0.5, 0.1, help="Kling range: 0.1–1.0 | Higher = stricter prompt following; 0.4–0.6 often best for natural motion")
    negative_prompt = st.text_area("Negative Prompt (avoid these)",
                                   value="blurry, low quality, artifacts, deformed, static, text, watermark, ugly, distorted, overexposed",
                                   height=100)

# ================== MAIN UI ==================
# ────────────────────────────────────────────────
#   Image Input Section (add this to your main UI)
# ────────────────────────────────────────────────
st.subheader("Provide Reference Image for Description")

input_method = st.radio(
    "Image source",
    options=["Upload local file", "Provide direct URL"],
    index=0,
    horizontal=True
)

image = None
image_url = None  # This will hold either remote URL or data: URL

if input_method == "Upload local file":
    uploaded_file = st.file_uploader(
        "Upload Nike shoe / athlete image (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

            # Convert to base64 data URL
            buffered = io.BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            st.error(f"Could not process uploaded image: {e}")

elif input_method == "Provide direct URL":
    url_input = st.text_input(
        "Image URL (public link, e.g. https://.../shoe.jpg)",
        placeholder="https://static.nike.com.hk/resources/product/IF0693-002/IF0693-002_VL1.png"
    )
    if url_input:
        try:
            # Optional: quick validation/fetch to show preview
            response = requests.head(url_input, timeout=5, allow_redirects=True)
            if response.status_code != 200:
                st.warning("URL may not be accessible (non-200 status). Try anyway?")
            st.image(url_input, caption="Preview from URL (if accessible)", use_column_width=True)
            image_url = url_input  # direct remote URL

        except Exception as e:
            st.error(f"Invalid or unreachable URL: {e}")

# ────────────────────────────────────────────────
#   Only proceed with prompt + generation if we have image input
# ────────────────────────────────────────────────
if image_url:
    prompt_text = st.text_area(
        "Instruction for cinematic marketing-style text descriptions",
        value="Describe this image generating detailed, cinematic marketing-style text descriptions. "
              "Focus on dynamic action, lighting, branding (e.g. swoosh, colors), camera movement, "
              "futuristic elements, high-energy sports commercial aesthetic.",
        height=120
    )
    if st.button("Generate Cinematic Marketing Description", type="primary"):
        with st.spinner("Calling Qwen vision model via API..."):
            try:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url   # ← this now works for BOTH remote URL and base64 data URL
                                    }
                                }
                            ]
                        }
                    ],
                    "model": "Qwen/Qwen3.5-397B-A17B:novita",  # or keep your original
                    # Optional: temperature, max_tokens, etc.
                    "temperature": 0.7,
                    "max_tokens": 500,
                }

                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()  # raise if 4xx/5xx

                result = response.json()
                # Adjust extraction based on actual response shape
                generated_text = result["choices"][0]["message"]["content"]

                st.success("Generated description:")
                st.markdown(generated_text)

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                if 'response' in locals():
                    st.json(response.text)  # debug raw response
            except KeyError as e:
                st.error(f"Unexpected response format: missing key {e}")
                st.json(result)

    if st.button("🚀 Generate Nike Promo Video (Kling 3.0 Pro)", type="primary"):
        with st.spinner("Encoding image + generating high-quality video on fal.ai... (1–5 minutes)"):
            try:
                # Convert PIL image to base64 data URL (JPEG for compatibility/size)
                buffered = io.BytesIO()
                image.convert("RGB").save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data_url = f"data:image/jpeg;base64,{img_base64}"

                # Run Kling 3.0 Pro I2V
                result = fal.subscribe(
                    "fal-ai/kling-video/v3/pro/image-to-video",
                    arguments={
                        "prompt": generated_text,
                        "start_image_url": image_data_url,  # fal.ai Kling accepts data URLs / base64
                        "duration": str(duration),          # Must be string
                        "aspect_ratio": aspect_ratio,
                        "negative_prompt": negative_prompt,
                        "cfg_scale": cfg_scale,
                        # Optional extras (uncomment if needed):
                        # "enable_audio": False,
                        # "mode": "professional",
                    }
                )

                # Extract video URL (handle different possible response shapes)
                video_url = None
                if isinstance(result, dict):
                    video_data = result.get("video", {})
                    video_url = video_data.get("url") if isinstance(video_data, dict) else result.get("video_url")
                elif isinstance(result, list) and result:
                    video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]

                if not video_url:
                    st.error("No valid video URL returned. Check fal.ai dashboard/logs.")
                    st.json(result)  # Debug output
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
else:
    st.info("Please provide a starting image using one of the options above to enable the prompt and generation button.")
