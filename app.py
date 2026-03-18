import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

st.set_page_config(page_title="Nike Video Generator (HF Inference)", layout="wide")

st.title("👟 Personalized Nike ReactX ZoomX Video Generator")
st.markdown("Upload your shoe image + customize → generate short motivational running clip via Hugging Face Inference")

# Sidebar for token & settings
with st.sidebar:
    st.header("Hugging Face Setup")
    st.markdown("[Get your token here](https://huggingface.co/settings/tokens) → create 'Inference Providers' token")
    hf_token = st.text_input("HF API Token", type="password")

    st.header("Generation Settings")
    provider = st.selectbox("Provider", ["fal-ai", "wavespeedai", "novita"], index=0)
    resolution = st.selectbox("Resolution", ["512x512", "720x480"], index=1)
    num_frames = st.slider("Frames (~seconds)", 25, 121, 81)

# Main content
uploaded_file = st.file_uploader("Upload your Nike shoe image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 2])

with col1:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your shoe", use_column_width=True)

with col2:
    name = st.text_input("Runner name", "Wei")
    city = st.text_input("City", "Shanghai")
    age = st.number_input("Age", 30, 50, 35)

    default_prompt = f"""Cinematic motivational running ad at dawn in {city}. 
    A confident {age}-year-old Chinese man named {name} runs smoothly along the riverside path. 
    He wears these exact white Nike ReactX ZoomX shoes with bold volt neon-green midsole glowing, black Swoosh, orange heel flash. 
    Dynamic tracking shots, slow-motion foot strikes with glowing energy return, sunrise lighting, empowering atmosphere, 
    fluid confident strides, professional Nike commercial style, high detail, vibrant colors."""

    prompt = st.text_area("Prompt", default_prompt, height=180)

if st.button("Generate Video", type="primary"):
    if not hf_token and not st.secrets.get("HF_TOKEN", None):
        st.error("Please provide your Hugging Face token (sidebar or secrets).")
    elif not uploaded_file:
        st.error("Please upload a shoe image first.")
    else:
        with st.spinner("Generating video... (usually 60–180 seconds)"):
            try:
                token = hf_token or st.secrets["HF_TOKEN"]
                client = InferenceClient(provider=provider, token=token)

                # Prepare image bytes
                img_bytes = io.BytesIO()
                Image.open(uploaded_file).save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Generate (using a supported I2V / T2V model - fal-ai often supports Wan variants)
                video_url = client.text_to_video(
                    prompt=prompt,
                    model="Wan-AI/Wan2.2-TI2V-5B",  # or "Lightricks/LTX-Video", "tencent/HunyuanVideo", etc.
                    image=img_bytes,
                    num_frames=num_frames,
                    height=480 if "480" in resolution else 720,
                    width=854 if "480" in resolution else 1280,
                    fps=16,
                )

                st.success("Video generated!")
                st.video(video_url)

                # Optional download
                st.download_button(
                    "Download MP4",
                    data=client.get_video_bytes(video_url),  # some clients support direct bytes
                    file_name=f"nike_{name}_{city}.mp4",
                    mime="video/mp4"
                )

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.info("""
Common fixes:
- Check token has Inference Providers permission
- Try different provider (fal-ai usually fastest for video)
- Use shorter num_frames or lower resolution
- Model may not support image input on this provider → remove image and retry
- Check https://huggingface.co/models?pipeline_tag=text-to-video for supported models/providers
                """)

st.markdown("---")
st.caption("Powered by Hugging Face Inference API • Pay-per-use (very cheap after free credits) • No local GPU needed")
