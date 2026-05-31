# ============================================================
# config.py — Central configuration for GAN-MNIST project
# All hyperparameters and paths defined here once and
# imported everywhere else — clean professional practice
# ============================================================

import os

# ── Base directory — works on any machine ────────────────────
# Gets the parent directory of this file (src/) then goes up
# one level to the project root
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Project paths ─────────────────────────────────────────────
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
RESULTS_DIR= os.path.join(BASE_DIR, 'results')
FAKE_DIR   = os.path.join(BASE_DIR, 'Fake_Digits')

# ── Create directories if they don't exist ───────────────────
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, RESULTS_DIR, FAKE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ── GAN Hyperparameters ───────────────────────────────────────
BATCH_SIZE = 100          # As specified in assignment
Z_DIM      = 100          # Latent vector dimensions
IMAGE_DIM  = 28 * 28      # MNIST image size flattened
LR         = 0.0002       # Learning rate for GAN
BETA1      = 0.5          # Adam beta1 — DCGAN standard
BETA2      = 0.999        # Adam beta2
GAN_EPOCHS = 50           # GAN training epochs

# ── Classifier Hyperparameters ────────────────────────────────
CLF_LR         = 0.001    # Classifier learning rate
CLF_BATCH_SIZE = 128      # Classifier batch size
CLF_EPOCHS     = 10       # Classifier training epochs

# ── Device ────────────────────────────────────────────────────
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model filenames ───────────────────────────────────────────
G_PATH = os.path.join(MODELS_DIR, 'G.pkl')
D_PATH = os.path.join(MODELS_DIR, 'D.pkl')
C_PATH = os.path.join(MODELS_DIR, 'C.pkl')