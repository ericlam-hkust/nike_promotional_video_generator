# app.py
import streamlit as st
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import traceback

st.set_page_config(page_title="Light Nike/PUMA Promo Video", layout="wide")
st.title("Lightweight Personalised Promo Video Generator")
st.markdown("Runs on free Streamlit Cloud (CPU only – very memory constrained)")

# ────────────────────────────────────────────────
#  LOAD MODELS – as light as possible
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading very light models… (this can take 1–3 min first time)")
def load_light_models():
    device = "cpu"
    dtype = torch.float32

    try:
        # 1. Very small BLIP for product caption
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        # 2. Tiny LLM – 0.5B version
        llm = pipeline(
            "text-generation",
            model="Qwen/Qwen2-0.5B-Instruct",
            torch_dtype=dtype,
            device=device,
            trust_remote_code=True
        )

        # 3. AnimateDiff Lightning – 4 step (fastest & lightest video model)
        adapter_repo = "ByteDance/AnimateDiff-Lightning"
        adapter_file = "animatediff_lightning_4step_diffusers.safetensors"

        adapter = MotionAdapter().to(device, dtype)
        state_dict = load_file(hf_hub_download(adapter_repo, adapter_file), device=device)
        adapter.load_state_dict(state_dict)

        pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=dtype,
        ).to(device)

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear"
        )

        return processor, blip_model, llm, pipe

    except Exception as e:
        st.error(f"Model loading failed:\n{str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None


processor, blip_model, llm_pipe, video_pipe = load_light_models()

if video_pipe is None:
    st.stop()

# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    uploaded_file = st.file_uploader("Upload shoe image", type=["jpg", "jpeg", "png"])

with col2:
    user_profile = st.text_area(
        "User description / profile",
        value="25-year-old trail runner, loves mountains, prefers bold colours",
        height=120
    )
    product_name = st.text_input("Product name (optional)", "PUMA Velocity Nitro")

if st.button("Generate short promo GIF", type="primary", disabled=not uploaded_file):
    if not user_profile.strip():
        st.warning("Please enter some user profile information")
        st.stop()

    with st.spinner("1/3 – Captioning product image …"):
        try:
            image = Image.open(uploaded_file).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(blip_model.device)
            out = blip_model.generate(**inputs, max_new_tokens=40)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            st.success(f"Detected: {caption}")
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            st.stop()

    with st.spinner("2/3 – Creating personalised prompt …"):
        try:
            template = f"""Create ONE short, dynamic marketing video prompt for this shoe: {caption} ({product_name}).
User: {user_profile}.
Style: energetic sports commercial, athlete wearing the shoe, cinematic, bold.
Output ONLY the prompt text. Max 60 words."""

            response = llm_pipe(
                template,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )[0]["generated_text"]

            # crude extraction
            prompt = response.split("Output ONLY")[-1].strip() if "Output ONLY" in response else response.strip()
            prompt = prompt[:250]  # safety limit
            st.caption("Generated prompt:")
            st.info(prompt)
        except Exception as e:
            st.error(f"Prompt generation failed: {e}")
            st.stop()

    with st.spinner("3/3 – Generating animated GIF (4-step Lightning – ~20–90 sec on CPU) …"):
        try:
            # Keep resolution low to save memory
            result = video_pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=1.0,
                num_frames=16,
                height=320,
                width=320
            )

            os.makedirs("output", exist_ok=True)
            gif_path = "output/promo.gif"

            export_to_gif(result.frames[0], gif_path)

            st.success("GIF created!")
            st.image(gif_path, caption="Personalised promo animation", use_column_width=True)

            with open(gif_path, "rb") as f:
                st.download_button(
                    label="Download GIF",
                    data=f,
                    file_name="personalised_puma_nike_promo.gif",
                    mime="image/gif"
                )

        except Exception as e:
            st.error("Video generation failed – most likely out of memory")
            st.error(str(e))
            if "memory" in str(e).lower() or "alloc" in str(e).lower():
                st.info("→ Try reducing height/width even more or use fewer frames (edit code).")
            st.error(traceback.format_exc())

st.markdown("---")
st.caption(
    "Very light stack: Qwen2-0.5B + BLIP-base + AnimateDiff-Lightning-4step\n"
    "Still memory-intensive – if it crashes → try Hugging Face Spaces (free CPU/GPU)"
)
