import streamlit as st
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from moviepy.editor import ImageSequenceClip

st.set_page_config(page_title="Nike/PUMA Light Personalised Video", layout="wide")
st.title("🎥 Lightweight Personalised Marketing Video (Runs on Streamlit Cloud Free)")
st.markdown("Product image + user profile → short promo video (AnimateDiff-Lightning 4-step)")

# ------------------- Load Light Models -------------------
@st.cache_resource
def load_models():
    # BLIP (same, very light)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
    
    # Light LLM for prompt personalisation
    llm_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2-1.5B-Instruct",
        torch_dtype=torch.float32,
        device="cpu",
        trust_remote_code=True
    )
    
    # AnimateDiff-Lightning (4-step – the lightest fast video model)
    step = 4
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    
    adapter = MotionAdapter().to("cpu", torch.float32)
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
    
    # Use a very light base model for lowest memory
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model, 
        motion_adapter=adapter, 
        torch_dtype=torch.float32
    ).to("cpu")
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear"
    )
    
    return blip_processor, blip_model, llm_pipeline, pipe

blip_processor, blip_model, llm_pipe, video_pipe = load_models()

# ------------------- UI -------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload product image (Nike/PUMA shoe)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Product Image", use_column_width=True)

with col2:
    user_profile = st.text_area(
        "User Profile",
        placeholder="25-year-old trail runner, loves mountains, bold colours, runs 3x/week",
        height=150
    )
    product_name = st.text_input("Product Name", "PUMA Velocity Nitro 3")

# ------------------- Generate -------------------
if st.button("🚀 Generate Light Personalised Video", type="primary") and uploaded_file and user_profile:
    with st.spinner("1/3 Captioning product..."):
        inputs = blip_processor(image, return_tensors="pt").to("cpu")
        caption_ids = blip_model.generate(**inputs, max_new_tokens=50)
        product_caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
        st.success(f"Product: {product_caption}")

    with st.spinner("2/3 Creating personalised prompt..."):
        prompt_template = f"""Create a short dynamic marketing video prompt for this shoe: {product_caption} ({product_name}).
Target user: {user_profile}.
Style: energetic Nike/PUMA commercial, athlete wearing the shoe, cinematic, bold colours.
Output ONLY the final prompt (max 70 words):"""
        
        llm_response = llm_pipe(prompt_template, max_new_tokens=120, temperature=0.7)[0]['generated_text']
        personalised_prompt = llm_response.split("Output ONLY")[-1].strip() if "Output ONLY" in llm_response else llm_response
        st.info(f"Prompt: {personalised_prompt[:150]}...")

    with st.spinner("3/3 Generating video (4-step Lightning – ~15-40 seconds on CPU)..."):
        output = video_pipe(
            prompt=personalised_prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            num_frames=16
        )
        
        # Save as GIF (fastest & lightest) and MP4
        os.makedirs("videos", exist_ok=True)
        gif_path = "videos/personalised_promo.gif"
        mp4_path = "videos/personalised_promo.mp4"
        
        export_to_gif(output.frames[0], gif_path)
        
        # Also create MP4 for better playback
        clip = ImageSequenceClip(output.frames[0], fps=8)
        clip.write_videofile(mp4_path, codec="libx264", fps=8, logger=None)
        
        st.success("✅ Video generated!")
        st.video(mp4_path)
        st.image(gif_path, caption="Animated GIF preview")
        st.download_button("📥 Download MP4", open(mp4_path, "rb"), "personalised_nike_puma_promo.mp4")
        st.download_button("📥 Download GIF", open(gif_path, "rb"), "personalised_promo.gif")

st.caption("Light stack: Qwen2-1.5B + BLIP + AnimateDiff-Lightning 4-step (all Hugging Face). Runs on CPU → perfect for free Streamlit Cloud!")
