# ============================================================
# streamlit_app.py — Interactive GAN demo
# Run with: streamlit run app/streamlit_app.py
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import streamlit as st
from PIL import Image

from src.config import DEVICE, Z_DIM, MODELS_DIR
from src.models import Generator, Classifier
from src.models import Generator_V2, Discriminator_V2, Classifier_V2

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="GAN MNIST Demo",
    page_icon="✏️",
    layout="centered"
)

# ============================================================
# Load models — cached so they only load once
# ============================================================
@st.cache_resource
def load_models():
    """Load G and C from models directory."""
    import pickle
    import io

    g_path = os.path.join(MODELS_DIR, 'G.pkl')
    c_path = os.path.join(MODELS_DIR, 'C.pkl')

    if not os.path.exists(g_path):
        st.error(f"Generator not found at {g_path}")
        st.stop()

    if not os.path.exists(c_path):
        st.error(f"Classifier not found at {c_path}")
        st.stop()

    # ── Custom unpickler that maps CUDA tensors to CPU ────────
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(
                    io.BytesIO(b),
                    map_location='cpu',
                    weights_only=False
                )
            return super().find_class(module, name)

    with open(g_path, 'rb') as f:
        G = CPUUnpickler(f).load()
    with open(c_path, 'rb') as f:
        C = CPUUnpickler(f).load()

    G = G.to(DEVICE)
    C = C.to(DEVICE)
    G.eval()
    C.eval()

    return G, C


# ============================================================
# Helper functions
# ============================================================
def generate_image(G, z=None, seed=None):
    """Generate a single image from G."""
    if seed is not None:
        torch.manual_seed(seed)
    if z is None:
        z = torch.randn(1, Z_DIM).to(DEVICE)

    with torch.no_grad():
        img = G(z).cpu().view(28, 28)
        img = (img + 1) / 2   # Denormalise to [0,1]

    img_np = (img.numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np).resize(
        (280, 280), Image.NEAREST
    )
    return pil_img, z


def classify_image(C, z):
    """Classify a generated image."""
    with torch.no_grad():
        # Generate image in classifier format [1, 1, 28, 28]
        img = C.__class__
        from src.config import DEVICE
        img_tensor = torch.zeros(1, 1, 28, 28).to(DEVICE)

        # Get the raw generated image
        G_ref = st.session_state.get('G')
        if G_ref is None:
            return None, None

        raw = G_ref(z).cpu().view(1, 1, 28, 28)
        raw = (raw + 1) / 2
        # Renormalise for classifier
        raw = (raw - 0.5) / 0.5
        raw = raw.to(DEVICE)

        outputs = C(raw)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred    = probs.argmax()

    return pred, probs


# ============================================================
# Main app
# ============================================================
def main():
    # ── Header ───────────────────────────────────────────────
    st.title("✏️ GAN MNIST Digit Generator")
    st.markdown(
        "A Generative Adversarial Network trained on MNIST "
        "generates realistic handwritten digits from random noise."
    )
    st.divider()

    # ── Load models ───────────────────────────────────────────
    G, C = load_models()
    st.session_state['G'] = G

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")

    use_seed = st.sidebar.checkbox("Fix random seed", value=False)
    seed_val = None
    if use_seed:
        seed_val = st.sidebar.slider(
            "Seed value", min_value=0, max_value=999, value=42
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "**Generator V2**  \n"
        "FC(100→256→512→1024→784)  \n"
        "BatchNorm + ReLU + Tanh  \n\n"
        "**Classifier V2**  \n"
        "3× Conv + BatchNorm  \n"
        "Dropout(0.5) + FC(10)  \n\n"
        "**Results**  \n"
        "D Accuracy: 52.99%  \n"
        "Classifier: 99.36%  \n"
        "S1 Error: 3.00%"
    )

    # ── Generate button ───────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button(
            "🎲 Generate New Digit",
            use_container_width=True,
            type="primary"
        )

    # ── Generate on button click or first load ────────────────
    if generate_btn or 'current_img' not in st.session_state:
        img, z = generate_image(G, seed=seed_val)
        st.session_state['current_img'] = img
        st.session_state['current_z']   = z

    img = st.session_state['current_img']
    z   = st.session_state['current_z']

    # ── Display image and classification ──────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Generated Image")
        st.image(img, caption="28×28 upscaled to 280×280",
                 use_container_width=True)

    with col2:
        st.markdown("### Classifier Prediction")

        # Classify
        with torch.no_grad():
            raw = G(z).cpu().view(1, 1, 28, 28)
            raw = (raw - 0.5) / 0.5
            raw = raw.to(DEVICE)
            outputs = C(raw)
            probs   = torch.softmax(
                outputs, dim=1
            ).cpu().numpy()[0]
            pred    = probs.argmax()

        st.metric(
            label="Predicted Digit",
            value=str(pred),
            delta=f"{probs[pred]*100:.1f}% confidence"
        )

        st.markdown("**Confidence per digit:**")
        for digit in range(10):
            st.progress(
                float(probs[digit]),
                text=f"Digit {digit}: {probs[digit]*100:.1f}%"
            )

    # ── Latent vector ─────────────────────────────────────────
    st.divider()
    st.markdown("### Latent Vector Z")
    st.markdown(
        "This is the 100-dimensional random noise vector "
        "that produced the image above."
    )

    z_np = z.cpu().numpy().flatten()
    st.code(
        f"z = [{', '.join([f'{v:.4f}' for v in z_np])}]",
        language=None
    )

    # ── Generate multiple ─────────────────────────────────────
    st.divider()
    st.markdown("### Generate Multiple Digits")
    st.markdown("Generate a grid of 25 random digits at once.")

    if st.button("🔢 Generate 25 Digits", type="secondary"):
        cols = st.columns(5)
        for i in range(25):
            img_i, _ = generate_image(G)
            with cols[i % 5]:
                st.image(
                    img_i,
                    caption=f"Sample {i+1}",
                    use_container_width=True
                )

    # ── Footer ────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "**P.A. Gunawardana** — University of Moratuwa  \n"
        "GitHub: [pas08/GAN-MNIST]"
        "(https://github.com/pas08/GAN-MNIST)"
    )


if __name__ == '__main__':
    main()