# ============================================================
# models.py — All neural network architectures
# Generator, Discriminator and Classifier defined here
# Imported by training scripts and Streamlit app
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# Generator V2 — Final architecture
# Input  : z vector [batch, 100]
# Output : fake image [batch, 784]
# ============================================================
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            # Layer 1: 100 → 256
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            # Layer 2: 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            # Layer 3: 512 → 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            # Output: 1024 → 784, Tanh maps to [-1,1]
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)


# ============================================================
# Discriminator V2 — Final architecture
# Input  : image [batch, 784]
# Output : probability [batch, 1]
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            # Layer 1: 784 → 512
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output: probability of being real
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


# ============================================================
# Classifier V2 — Final architecture
# Input  : image [batch, 1, 28, 28]
# Output : class scores [batch, 10]
# ============================================================
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Conv block 1: [b,1,28,28] → [b,32,14,14]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Conv block 2: [b,32,14,14] → [b,64,7,7]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Conv block 3: [b,64,7,7] → [b,128,3,3]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected: 1152 → 256 → 10
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


# ============================================================
# Latent Mapper — for conditional generation (Phase 4)
# Input  : one-hot digit class [batch, 10]
# Output : latent vector z [batch, 100]
# ============================================================
class LatentMapper(nn.Module):
    def __init__(self, num_classes=10, z_dim=100):
        super(LatentMapper, self).__init__()

        self.mapper = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mapper(x)


# ============================================================
# Quick architecture test
# ============================================================
if __name__ == '__main__':
    import torch

    # Test Generator
    G = Generator()
    z = torch.randn(4, 100)
    out = G(z)
    print(f"Generator   : {z.shape} → {out.shape}")

    # Test Discriminator
    D = Discriminator()
    img = torch.randn(4, 784)
    out = D(img)
    print(f"Discriminator: {img.shape} → {out.shape}")

    # Test Classifier
    C = Classifier()
    img = torch.randn(4, 1, 28, 28)
    out = C(img)
    print(f"Classifier  : {img.shape} → {out.shape}")

    # Test LatentMapper
    M = LatentMapper()
    one_hot = torch.zeros(4, 10)
    one_hot[0][7] = 1
    out = M(one_hot)
    print(f"LatentMapper: {one_hot.shape} → {out.shape}")

    print("\nAll architectures verified.")
    
# ============================================================
# Class aliases — required for loading pkl files saved
# in Colab where classes were named Generator_V2,
# Discriminator_V2 and Classifier_V2
# ============================================================
Generator_V2     = Generator
Discriminator_V2 = Discriminator
Classifier_V2    = Classifier