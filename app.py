import streamlit as st
import torch
from diffusers import CogVideoXImageToVideoPipeline
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

st.set_page_config(page_title="Nike/PUMA Personalised Video Generator", layout="wide")
st.title("🎥 Personalised Marketing Video Generator (Nike / PUMA)")
st.markdown("Upload a product image + user profile → get a custom promotional video")

# ------------------- Load Models (cached) -------------------
@st.cache_resource
def load_models():
    # 1. BLIP for product captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. LLM for personalised prompt
    llm_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Video model - CogVideoX-5b-I2V (best I2V pipeline)
    video_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "zai-org/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16,
        variant="bf16"
    )
    video_pipe.enable_model_cpu_offload()  # saves VRAM
    video_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    return blip_processor, blip_model, llm_pipeline, video_pipe

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
        "User Profile (membership or social media)",
        placeholder="25-year-old male trail runner, loves mountains, prefers bold colours, runs 3x/week",
        height=150
    )
    product_name = st.text_input("Product Name (optional)", "PUMA Velocity Nitro 3")

# ------------------- Generate Button -------------------
if st.button("🚀 Generate Personalised Promotional Video", type="primary") and uploaded_file and user_profile:
    with st.spinner("1/3 Captioning product..."):
        # BLIP caption
        inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
        caption = blip_model.generate(**inputs, max_new_tokens=50)
        product_caption = blip_processor.decode(caption[0], skip_special_tokens=True)
        st.success(f"Product detected: {product_caption}")

    with st.spinner("2/3 Creating personalised prompt..."):
        # LLM personalisation
        prompt_template = f"""Create a short, dynamic promotional video prompt for this shoe: {product_caption} ({product_name}).
Target user: {user_profile}.
Style: cinematic Nike/PUMA marketing video, energetic music vibe, slow-motion highlights, bold colours, athlete wearing the shoe naturally.
Output ONLY the final video generation prompt (max 80 words):"""
        
        llm_response = llm_pipe(
            prompt_template,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )[0]['generated_text']
        
        # Extract the final prompt
        personalised_prompt = llm_response.split("Output ONLY")[-1].strip() if "Output ONLY" in llm_response else llm_response
        st.info(f"Personalised prompt: {personalised_prompt[:200]}...")

    with st.spinner("3/3 Generating video (this takes 3–8 minutes on GPU)..."):
        # Generate video with CogVideoX I2V
        video_frames = video_pipe(
            image=image.resize((512, 512)),  # model expects ~512px
            prompt=personalised_prompt,
            num_frames=16,          # ~5–8 seconds at 24fps
            num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(42)
        ).frames[0]
        
        # Save as MP4
        os.makedirs("videos", exist_ok=True)
        output_path = "videos/personalised_promo.mp4"
        
        # Simple frame → video (uses moviepy or OpenCV – install if needed)
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(video_frames, fps=8)  # low fps for demo speed
        clip.write_videofile(output_path, codec="libx264", fps=8)
        
        st.success("✅ Video generated!")
        st.video(output_path)
        st.download_button("📥 Download Video", data=open(output_path, "rb"), file_name="personalised_puma_nike_promo.mp4")

st.caption("Models used: Qwen2-7B-Instruct + BLIP + CogVideoX-5b-I2V (all on Hugging Face). Requires GPU with ≥16GB VRAM.")
