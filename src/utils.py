# ============================================================
# utils.py — Utility functions used across the project
# Model saving/loading, image generation, visualisation
# ============================================================

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from src.config import (
    DEVICE, Z_DIM, MODELS_DIR, OUTPUT_DIR,
    G_PATH, D_PATH, C_PATH
)


# ============================================================
# Model saving and loading
# ============================================================

def save_model(model, path):
    """Save a model using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path, model_class=None):
    """
    Load a model from a pickle file.
    model_class must be imported before calling this
    so pickle can find the class definition.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded ← {path}")
    return model


def load_all_models():
    """Load G, D and C from the models directory."""
    # Import here so pickle can find class definitions
    from src.models import Generator, Discriminator, Classifier
    G = load_model(G_PATH)
    D = load_model(D_PATH)
    C = load_model(C_PATH)
    return G, D, C


def load_generator():
    """Load only Generator G — used in Streamlit app."""
    from src.models import Generator
    return load_model(G_PATH)


def load_classifier():
    """Load only Classifier C — used in evaluation."""
    from src.models import Classifier
    return load_model(C_PATH)


# ============================================================
# Image generation
# ============================================================

def generate_images(G, n=100, seed=None):
    """
    Generate n fake images using Generator G.
    Returns images as tensor [n, 28, 28] in range [0,1]
    and corresponding z vectors [n, 100].
    """
    if seed is not None:
        torch.manual_seed(seed)

    G.eval()
    with torch.no_grad():
        z = torch.randn(n, Z_DIM).to(DEVICE)
        imgs = G(z).cpu().view(n, 28, 28)
        imgs = (imgs + 1) / 2   # Denormalise to [0,1]
    return imgs, z.cpu()


def generate_single(G, z=None):
    """
    Generate a single image from Generator G.
    If z is None, uses random noise.
    Returns PIL image and z vector.
    """
    G.eval()
    with torch.no_grad():
        if z is None:
            z = torch.randn(1, Z_DIM).to(DEVICE)
        img = G(z).cpu().view(28, 28)
        img = (img + 1) / 2
        img_np = (img.numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
    return pil_img, z.cpu()


# ============================================================
# Visualisation
# ============================================================

def save_grid(imgs, nrow=10, title='Generated Digits',
              filename='grid.png'):
    """
    Save a grid of images to the outputs folder.
    imgs: tensor [n, 28, 28] in range [0,1]
    """
    imgs_4d = imgs.unsqueeze(1)   # [n, 1, 28, 28]
    grid = torchvision.utils.make_grid(
        imgs_4d, nrow=nrow, padding=2, normalize=False
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(title)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Grid saved → {save_path}")
    return save_path


def plot_training_curves(g_losses, d_losses, d_accs,
                         filename='training_curves.png'):
    """Plot and save GAN training curves."""
    epochs = range(1, len(g_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, g_losses, label='Generator Loss',
             color='blue', linewidth=2)
    ax1.plot(epochs, d_losses, label='Discriminator Loss',
             color='red', linewidth=2)
    ax1.axhline(y=0.693, color='gray', linestyle='--',
                alpha=0.7, label='Ideal D Loss (0.693)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('GAN Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, d_accs, label='D Accuracy',
             color='green', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--',
                alpha=0.7, label='Ideal (50%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Discriminator Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")
    return save_path


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    from src.models import Generator

    print("Testing image generation...")
    G = Generator().to(DEVICE)
    imgs, z = generate_images(G, n=100, seed=42)
    print(f"Generated images shape : {imgs.shape}")
    print(f"Z vectors shape        : {z.shape}")
    print(f"Pixel range            : [{imgs.min():.3f}, "
          f"{imgs.max():.3f}]")

    save_grid(imgs, title='Test Grid', filename='test_grid.png')
    print("\nutils.py verified.")