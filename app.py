import streamlit as st
import replicate
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Nike ReactX ZoomX Video Generator", page_icon="👟", layout="centered")

st.title("🚀 Nike ReactX ZoomX – Personalized Video Generator")
st.markdown("**Your 35-year-old Chinese runner story comes to life** — powered by Wan 2.2 I2V Fast")

# === Sidebar ===
with st.sidebar:
    st.header("Your Replicate API Key")
    st.markdown("Get free credits at [replicate.com](https://replicate.com) → Account → API")
    api_key = st.text_input("REPLICATE_API_TOKEN", type="password", help="Stored only in your Streamlit secrets")
    
    st.header("Settings")
    resolution = st.selectbox("Resolution", ["480p", "720p"], index=0)
    go_fast = st.checkbox("Go Fast (recommended)", value=True)
    num_frames = st.slider("Number of frames", 81, 121, 81)

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

# Pre-filled powerful prompt (customized from our storyline)
default_prompt = f"Cinematic motivational running ad at dawn in {city}. A confident {age}-year-old Chinese man named {name} runs smoothly along the riverside path. He wears these exact white Nike ReactX ZoomX shoes with bold volt neon-green midsole, black Swoosh, and orange heel flash. Dynamic tracking shots, slow-motion foot strikes with glowing energy return, sunrise lighting, empowering atmosphere, fluid confident strides, professional Nike commercial style."

prompt = st.text_area("Prompt (auto-generated but editable)", default_prompt, height=150)

if st.button("🎥 Generate Video", type="primary", use_container_width=True):
    if not api_key and not st.secrets.get("REPLICATE_API_TOKEN"):
        st.error("Please add your Replicate API key in the sidebar or in Streamlit secrets.")
    elif not uploaded_image:
        st.error("Please upload your Nike shoe image first.")
    else:
        with st.spinner("Generating your personalized Nike video... (usually 45–90 seconds)"):
            try:
                # Use secret or sidebar key
                token = api_key or st.secrets["REPLICATE_API_TOKEN"]
                client = replicate.Client(api_token=token)
                
                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_image.getvalue())
                    image_path = tmp.name
                
                input_params = {
                    "prompt": prompt,
                    "image": open(image_path, "rb"),
                    "num_frames": num_frames,
                    "resolution": resolution,
                    "frames_per_second": 16,
                    "go_fast": go_fast,
                    "interpolate_to_30fps": True,
                    "sample_shift": 12,
                    "disable_safety_checker": False
                }
                
                output = client.run(
                    "wan-video/wan-2.2-i2v-fast",
                    input=input_params
                )
                
                # Clean up temp file
                os.unlink(image_path)
                
                st.success("✅ Video generated successfully!")
                st.video(output)  # Replicate returns direct mp4 URL
                
                st.download_button(
                    label="📥 Download Video",
                    data=open(output, "rb").read() if isinstance(output, str) else output,
                    file_name=f"nikexzoomx_{name}_{city}.mp4",
                    mime="video/mp4"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Tips: Check your API key has credits. Try 480p + Go Fast for quicker/cheaper results.")

st.caption("Model: wan-video/wan-2.2-i2v-fast • ~$0.05 per video • Powered by Replicate")
