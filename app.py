import streamlit as st
import fal_client as fal
import requests
import tempfile
import os
from PIL import Image
import base64
import io
from huggingface_hub import InferenceClient
from pathlib import Path

def normalize_video_output(output):
    if output is None:
        return None

    if isinstance(output, (str, Path)):
        s = str(output)
        if s.startswith("http://") or s.startswith("https://"):
            return s
        if Path(s).exists():
            return s
        return None

    if isinstance(output, bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(output)
        tmp.close()
        return tmp.name

    if hasattr(output, "read"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(output.read())
        tmp.close()
        return tmp.name

    if hasattr(output, "url"):
        return output.url

    if isinstance(output, dict):
        for k in ["url", "video_url", "file", "path"]:
            v = output.get(k)
            if isinstance(v, str) and v:
                return v
        data = output.get("video")
        if isinstance(data, bytes):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(data)
            tmp.close()
            return tmp.name

    return None

# Set FAL_KEY from secrets (fal-client reads from env)
if "FAL_KEY" in st.secrets:
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
else:
    st.error("❌ FAL_KEY not found in .streamlit/secrets.toml or Streamlit app secrets.")
    st.stop()

st.set_page_config(page_title="Nike Video Generator", page_icon="🏃", layout="wide")
st.title("🏃 Nike Commercial Video Generator")
st.subheader("High-quality Image-Text-to-Video • No local GPU!")
st.caption("Cinematic motion, fluid dynamics, strong prompt adherence | Up to 15s clips | Commercial Use")

# ================== SIDEBAR SETTINGS ==================
with st.sidebar:
    st.header("👤 User Profile")
    st.caption("This personalizes your Nike promo (embedded in prompt)")

    user_name = st.text_input("Name", value="Eric", max_chars=50)
    user_age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)
    user_gender = st.selectbox("Gender", 
                               options=["Man", "Woman", "Non-binary", "Prefer not to say"],
                               index=0)
    user_city = st.selectbox("City / District (HK)", 
                             options=[
                                 "Hong Kong Island", "Kowloon", "New Territories",
                                 "Central and Western", "Wan Chai", "Eastern", "Southern",
                                 "Yau Tsim Mong", "Sham Shui Po", "Kowloon City",
                                 "Tsuen Wan", "Sha Tin", "Tuen Mun", "Other"
                             ],
                             index=0)
    user_race = st.selectbox("Ethnicity / Race",
                             options=[
                                 "Chinese", "Filipino", "Indonesian", "South Asian (Indian/Pakistani/Nepalese)",
                                 "White", "Other Asian", "Other", "Prefer not to say"
                             ],
                             index=0)
    user_language = st.selectbox("Language", options=["English"], index=0, disabled=True)

    st.header("⚙️ Generation Settings")
    # st.info("Model: fal-ai/kling-video/v3/pro/image-to-video\n"
    #        "Output: Up to 1080p native in Pro mode\n"
    #        "Generation: ~60–300 seconds | Cost: ~$0.10–0.30 per clip")
   
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
            st.image(image, caption="Uploaded image", width=400)

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
            st.image(url_input, caption="Preview from URL (if accessible)", width=400)
            image_url = url_input  # direct remote URL

        except Exception as e:
            st.error(f"Invalid or unreachable URL: {e}")
else:
    st.info("Please provide a starting image using one of the options above to enable the prompt and generation button.")

# ────────────────────────────────────────────────
#   Only proceed with prompt + generation if we have image input
# ────────────────────────────────────────────────
if "generated_text" not in st.session_state:
    st.session_state.generated_text = None
if image_url:
    system_prompt = st.text_area(
        "Base Marketing / Motion Prompt",
        value=f"""You are a world-class cinematic prompt engineer and Nike advertising creative director.
            
            Your job is to analyze the provided image and generate ONE single, extremely detailed, ready-to-use text prompt for high-end image-text-to-video model.
            
            You MUST follow this exact layered structure and style (never deviate):
            
            [Subject / Hero Shot]: 
            [Scene & Environment]: 
            [Motion & Dynamics]: 
            [Camera & Cinematography]: 
            [Lighting & Mood]: 
            [Personalization Layer]: 
            [Style & Quality Boosters]:
            
            Use highly vivid, professional advertising language with cinematic terms (tracking pan, dolly zoom, orbiting crane shot, low-angle side-to-front reveal, anamorphic lenses, ARRI Alexa 65, 60fps slow-motion bursts, volumetric god rays, lens flares, etc.).
            
            Emphasize visible Swoosh branding on clothing and billboards, high-energy athletic motion, futuristic neon city at golden hour sunset, motivational and empowering atmosphere.
            
            Personalization is CRITICAL: Tailor the energy, tone, appeal, and cultural resonance specifically for a {user_age}-year-old {user_gender} named {user_name} from {user_city}, {user_race} background. Make it feel empowering and perfectly matched to this demographic.
            
            Output ONLY the final prompt text — nothing else. No explanations, no JSON, no markdown, no extra words. Start directly with "[Subject / Hero Shot]:".""",
        height=150,
        help="This will be combined with user profile for personalization."
    )
    full_user_message = f"""Analyze this image in extreme detail and create the perfect cinematic Nike commercial video prompt.
                        User profile for personalization:
                        - Name: {user_name}
                        - Age: {user_age}
                        - Gender: {user_gender}
                        - City: {user_city}
                        - Ethnicity: {user_race}
                        
                        Generate the prompt now."""

    if st.button("🚀 Generate Cinematic Marketing-style Script", type="primary"):
        with st.spinner("Calling Qwen3.5-35B-A3B model via API..."):
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},  # or base64 data URL
                                {"type": "text", "text": full_user_message}
                            ]
                        }
                    ],
                    "model": "Qwen/Qwen3.5-397B-A17B:novita",  # or keep your original
                    # Optional: temperature, max_tokens, etc.
                    "temperature": 0.7
                    # "max_tokens": 500,
                }
                API_URL = "https://router.huggingface.co/v1/chat/completions"
                os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
                headers = {
                    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                }
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()  # raise if 4xx/5xx

                result = response.json()
                # Adjust extraction based on actual response shape
                st.session_state.generated_text = result["choices"][0]["message"]["content"]

                st.subheader("Generated Script:")
                st.markdown(st.session_state.generated_text)

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                if 'response' in locals():
                    st.json(response.text)  # debug raw response
            except KeyError as e:
                st.error(f"Unexpected response format: missing key {e}")
                st.json(result)
    
if st.session_state.generated_text:
    # --- Model selector ---
    model_choice = st.radio(
        "Choose video model",
        options=[
            "FREE MODEL: Wan-AI/Wan2.2-I2V-A14B (Hugging Face)",
            "PAID MODEL: Kling 3.0 Pro (fal.ai)"
        ],
        index=0,
        horizontal=True,
    )

    if st.button("🚀 Generate High-Quality Promo Video", type="primary"):
        with st.spinner("Encoding image + generating high-quality video... (1–5 minutes)"):
            st.subheader("Generated Script:")
            st.markdown(st.session_state.generated_text)
            try:
                buffered = io.BytesIO()
                image.convert("RGB").save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data_url = f"data:image/jpeg;base64,{img_base64}"
    
                video_source = None
    
                if model_choice == "PAID MODEL: Kling 3.0 Pro (fal.ai)":
                    result = fal.subscribe(
                        "fal-ai/kling-video/v3/pro/image-to-video",
                        arguments={
                            "prompt": st.session_state.generated_text,
                            "start_image_url": image_data_url,
                            "duration": str(duration),
                            "aspect_ratio": aspect_ratio,
                            "negative_prompt": negative_prompt,
                            "cfg_scale": cfg_scale,
                            "mode": "professional"
                        }
                    )
    
                    video_source = normalize_video_output(result)
    
                elif model_choice == "FREE MODEL: Wan-AI/Wan2.2-I2V-A14B (Hugging Face)":
                    client = InferenceClient(
                        provider="fal-ai",
                        api_key=st.secrets["HF_TOKEN"],
                    )
    
                    segments = []
                    for i in range(3):  # 3 segments = 15s
                        prompt_seg = f"{st.session_state.generated_text} [segment {i+1}/3]"
                        video_seg = client.image_to_video(
                            image=image_data_url if i == 0 else segments[-1],  # use previous segment as start image
                            prompt=prompt_seg,
                            model="Wan-AI/Wan2.2-I2V-A14B",
                        )
                        segments.append(normalize_video_output(video_seg))
                    
                    # Stitch with moviepy or ffmpeg
                    final_video = stitch_videos(segments)
                    st.session_state.video_source = final_video
    
                if not video_source:
                    st.error("No playable video output returned.")
                    st.write("Returned type:", type(result) if "result" in locals() else type(video))
                    st.stop()
    
                st.session_state.video_source = video_source
    
            except Exception as e:
                st.error(f"Video generation failed: {str(e)}")
                st.stop()
    
            if st.session_state.get("video_source"):
                st.success(f"✅ High-quality Nike commercial video generated with {model_choice}")
                st.video(st.session_state.video_source)


        # Download button
        try:
            with open(st.session_state.video_source, "rb") as f:
                st.download_button(
                    label="📥 Download MP4 for your project",
                    data=f.read(),
                    file_name=f"nike_promo_{duration}s.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.warning("Video playable above — right-click player to save if download fails.")
