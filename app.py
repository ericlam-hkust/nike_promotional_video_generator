import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import tempfile
import os
import io

st.set_page_config(page_title="Nike ReactX ZoomX Video Generator (HF)", page_icon="👟", layout="centered")

st.title("🚀 Nike ReactX ZoomX – Personalized Video Generator (Hugging Face)")
st.markdown("Powered by Wan 2.2 I2V via Hugging Face Inference Providers")

# === Sidebar ===
with st.sidebar:
    st.header("Your Hugging Face API Token")
    st.markdown("Create at https://huggingface.co/settings/tokens (enable 'Inference Providers')")
    hf_token = st.text_input("HF_TOKEN", type="password", help="Stored only in secrets for deployed app")
    
    st.header("Settings")
    provider = st.selectbox("Inference Provider", ["fal-ai", "wavespeedai"], index=0)  # fal-ai often fastest/cheapest for video
    resolution = st.selectbox("Resolution (if supported)", ["480p", "720p"], index=0)

# === Main UI ===
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_image = st.file_uploader("Upload your Nike shoe image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Your shoe", use_column_width=True)

with col2:
    name = st.text_input("Runner's Name", value="Wei")
    city = st.text_input("City", value="Shanghai")
    age = st.number_input("Age", value=35, min_value=30, max_value=50)

# Auto-generated prompt (same as before)
default_prompt = f"Cinematic motivational running ad at dawn in {city}. A confident {age}-year-old Chinese man named {name} runs smoothly along the riverside path. He wears these exact white Nike ReactX ZoomX shoes with bold volt neon-green midsole, black Swoosh, and orange heel flash. Dynamic tracking shots, slow-motion foot strikes with glowing energy return, sunrise lighting, empowering atmosphere, fluid confident strides, professional Nike commercial style."

prompt = st.text_area("Prompt (editable)", default_prompt, height=150)

if st.button("🎥 Generate Video", type="primary", use_container_width=True):
    if not hf_token and not st.secrets.get("HF_TOKEN"):
        st.error("Please add your Hugging Face token in the sidebar or in Streamlit secrets.")
    elif not uploaded_image:
        st.error("Please upload your Nike shoe image first.")
    else:
        with st.spinner("Generating video via Hugging Face Inference... (60–180 seconds depending on queue)"):
            try:
                token = hf_token or st.secrets["HF_TOKEN"]
                client = InferenceClient(provider=provider, token=token)
                
                # Save image temporarily (InferenceClient accepts file-like or path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_image.getvalue())
                    image_path = tmp.name
                
                # Call text-to-video or image-to-video (Wan 2.2 supports I2V)
                # Model: Use a Wan variant that supports I2V via your provider (fal-ai hosts many)
                video = client.text_to_video(
                    prompt=prompt,
                    model="Wan-AI/Wan2.2-TI2V-5B",   # Or "Wan-AI/Wan2.2-I2V-..." if exact match; check model card
                    image=open(image_path, "rb"),    # For image-to-video conditioning
                    # Optional params (provider-dependent; fal-ai supports these)
                    num_frames=81,
                    height=480 if resolution == "480p" else 720,
                    width=848 if resolution == "480p" else 1280,  # Approx 16:9
                    fps=16,
                    # Add more if provider docs list them (e.g. guidance_scale, num_inference_steps)
                )
                
                os.unlink(image_path)
                
                # video is bytes → display directly
                st.success("✅ Video generated!")
                st.video(video)
                
                # Download button
                st.download_button(
                    label="📥 Download Video",
                    data=video,
                    file_name=f"nikexzoomx_{name}_{city}.mp4",
                    mime="video/mp4"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("""
Tips:
- Check token has 'Inference Providers' permission.
- fal-ai is usually fastest/cheapest for Wan models.
- If model not found, try: HunyuanVideo, CogVideoX, or search https://huggingface.co/models?pipeline_tag=text-to-video&other=video-generation
- First generations may be slow due to cold start.
                """)

st.caption("Model: Wan-AI/Wan2.2-TI2V-5B (or similar) • HF Inference Providers • Pay-per-use (~$0.02–$0.10/video)")
