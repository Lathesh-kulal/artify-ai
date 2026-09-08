import streamlit as st
from PIL import Image
import os
import io
import torch

from core.style_transfer import apply_style_transfer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Artify AI - Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS for Modern Dark & Glassmorphism Aesthetic
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Hero Header Gradient */
    .hero-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #A064FF 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.15rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    /* Style Badge */
    .style-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(160, 100, 255, 0.15);
        color: #C084FC;
        border: 1px solid rgba(160, 100, 255, 0.3);
    }

    /* Metric Badges */
    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Device & Engine Check
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
device_display = "⚡ CUDA (GPU) Accelerated" if device == "cuda" else "🖥️ CPU Mode"

# --------------------------------------------------
# Sidebar: Controls & Options
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Studio Settings")
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;">
        <span style="font-size: 0.82rem; color: #94A3B8;">PyTorch Compute Engine:</span><br/>
        <strong style="color: {'#4ADE80' if device == 'cuda' else '#38BDF8'}; font-size: 0.95rem;">{device_display}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Transformation Controls")
    preserve_color = st.toggle(
        "Preserve Original Colors",
        value=False,
        help="Keep the color palette of the input photo while applying the artistic style texture."
    )
    
    quality_option = st.select_slider(
        "Max Resolution",
        options=["Fast (512px)", "Standard (768px)", "HD (1024px)"],
        value="Standard (768px)"
    )
    dim_map = {
        "Fast (512px)": 512,
        "Standard (768px)": 768,
        "HD (1024px)": 1024
    }
    max_dim = dim_map[quality_option]

    style_intensity = st.slider(
        "Style Intensity",
        min_value=0.2,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Blend factor between original photo and stylized output (1.0 = pure style)."
    )

    st.divider()

    st.markdown("#### 🖼️ Quick Demo Presets")
    use_sample = st.checkbox("Use Demo Photo (Golden Retriever)", value=False)

    st.divider()

    st.markdown("#### ℹ️ About Artify AI")
    st.caption(
        "Artify AI utilizes Feed-Forward Convolutional Neural Networks (Johnson et al.) "
        "trained on perceptual loss functions for high-speed style synthesis."
    )

# --------------------------------------------------
# Hero Header
# --------------------------------------------------
st.markdown('<div class="hero-title">🎨 Artify AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Transform your ordinary photos into museum-grade AI artwork in seconds.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Style Catalog
# --------------------------------------------------
STYLES_CATALOG = {
    "Candy": {
        "tag": "Colorful Pop Art",
        "description": "Vibrant, swirling chromatic textures inspired by modern pop confectionery aesthetics.",
        "icon": "🍬"
    },
    "Mosaic": {
        "tag": "Byzantine Art",
        "description": "Segmented tile motifs with intricate structural patterns reminiscent of classical stained glass.",
        "icon": "🧩"
    },
    "Rain Princess": {
        "tag": "Impressionism",
        "description": "Dramatic, colorful brush strokes with deep contrast and dynamic light dispersion by Leonid Afremov.",
        "icon": "🌧️"
    },
    "Starry Night": {
        "tag": "Post-Impressionism",
        "description": "Vincent van Gogh's legendary swirling nocturnal skies, luminous stars, and vivid cobalt blues.",
        "icon": "✨"
    },
    "Udnie": {
        "tag": "Cubist Futurism",
        "description": "Francis Picabia's abstract geometric rhythm, sharp energetic angles, and multi-dimensional planes.",
        "icon": "🎭"
    }
}

# --------------------------------------------------
# Studio Layout
# --------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

# --------------------------------------------------
# Left Column: Upload & Input Image
# --------------------------------------------------
with col1:
    st.markdown("### 1. 📤 Upload Content Photo")
    
    content_image = None

    if use_sample:
        sample_path = os.path.join("assets", "samples", "sample_dog.jpg")
        if os.path.exists(sample_path):
            content_image = Image.open(sample_path).convert("RGB")
            st.info("Using built-in demo photo (Golden Retriever in Park).")
            st.image(content_image, caption=f"Sample Image ({content_image.width} × {content_image.height} px)", width="stretch")
    else:
        uploaded_file = st.file_uploader(
            "Drop your photo here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supports high-resolution portraits, nature, or architecture photos."
        )

        if uploaded_file is not None:
            try:
                content_image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    content_image,
                    caption=f"Original Photo ({content_image.width} × {content_image.height} px)",
                    width="stretch"
                )
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 45px 20px; border: 2px dashed rgba(255,255,255,0.15);">
                <div style="font-size: 2.6rem; margin-bottom: 8px;">📸</div>
                <div style="color: #94A3B8; font-size: 0.95rem;">
                    Upload any photo here to begin,<br/>
                    or toggle <strong>'Use Demo Landscape Image'</strong> in the sidebar.
                </div>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# Right Column: Style Selection & Neural Transformation
# --------------------------------------------------
with col2:
    st.markdown("### 2. 🎭 Choose Master Style")
    
    selected_style_name = st.selectbox(
        "Select an artistic movement:",
        options=list(STYLES_CATALOG.keys()),
        format_func=lambda x: f"{STYLES_CATALOG[x]['icon']} {x} — {STYLES_CATALOG[x]['tag']}"
    )

    style_info = STYLES_CATALOG[selected_style_name]
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; font-size: 1.15rem; color: #F1F5F9;">{style_info['icon']} {selected_style_name}</span>
            <span class="style-pill">{style_info['tag']}</span>
        </div>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 8px; line-height: 1.4;">
            {style_info['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 3. 🚀 Transform")
    generate_btn = st.button("✨ Transform into Artwork", width="stretch", type="primary")

    # Session state for output storage across reruns
    if "result_image" not in st.session_state:
        st.session_state.result_image = None
        st.session_state.elapsed_time = 0.0
        st.session_state.last_style = ""

    if generate_btn:
        if content_image is None:
            st.warning("⚠️ Please upload a photo or check 'Use Demo Photo (Golden Retriever)' first!")
        else:
            with st.spinner(f"Rendering {selected_style_name} style via Neural Network..."):
                try:
                    stylized_img, elapsed = apply_style_transfer(
                        content_image=content_image,
                        style_name=selected_style_name,
                        device=device,
                        max_dim=max_dim,
                        preserve_color=preserve_color,
                        style_intensity=style_intensity
                    )
                    st.session_state.result_image = stylized_img
                    st.session_state.elapsed_time = elapsed
                    st.session_state.last_style = selected_style_name
                    st.toast(f"Artwork created in {elapsed:.2f}s!", icon="🎉")
                except Exception as e:
                    st.error(f"Failed to generate artwork: {e}")

# --------------------------------------------------
# Display Artwork Result (Full Width or Columns)
# --------------------------------------------------
if st.session_state.result_image is not None and content_image is not None:
    st.divider()
    st.markdown(f"### 🖼️ Masterpiece Generated: *{st.session_state.last_style}*")
    
    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 12px;">
        <span class="stat-badge">⏱️ Rendered in {st.session_state.elapsed_time:.2f}s</span>
        <span class="stat-badge">📐 {st.session_state.result_image.width} × {st.session_state.result_image.height} px</span>
        <span class="stat-badge">🎨 Style: {st.session_state.last_style}</span>
    </div>
    """, unsafe_allow_html=True)

    out_col1, out_col2 = st.columns(2, gap="medium")
    with out_col1:
        st.image(content_image, caption="Original Photo", width="stretch")
    with out_col2:
        st.image(st.session_state.result_image, caption=f"Stylized Artwork ({st.session_state.last_style})", width="stretch")

    # Download Button
    buf = io.BytesIO()
    st.session_state.result_image.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()

    filename = f"artify_{st.session_state.last_style.lower().replace(' ', '_')}.jpg"
    st.download_button(
        label=f"💾 Download Artwork ({filename})",
        data=byte_im,
        file_name=filename,
        mime="image/jpeg",
        width="stretch"
    )

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;">
    Crafted with ❤️ for Artify AI • Neural Style Transfer Studio • Powered by Streamlit & PyTorch
</div>
""", unsafe_allow_html=True)
