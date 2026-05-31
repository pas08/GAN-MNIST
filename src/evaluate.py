# ============================================================
# evaluate.py — Evaluation functions for S0 and S1
# Computes classification errors on real and fake datasets
# ============================================================

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import (
    DEVICE, DATA_DIR, FAKE_DIR,
    BATCH_SIZE, CLF_BATCH_SIZE
)


# ============================================================
# Data loading
# ============================================================

def get_mnist_test_loader(batch_size=CLF_BATCH_SIZE):
    """Load full MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=True
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    ), test_dataset


def get_s1_loader():
    """
    Load fake digit dataset S1 using ImageFolder.
    Folder structure: Fake_Digits/<label>/<index>.png
    As hinted in the assignment.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    s1_dataset = ImageFolder(
        root=FAKE_DIR,
        transform=transform
    )
    s1_loader = DataLoader(
        s1_dataset,
        batch_size=100,
        shuffle=False
    )
    return s1_loader, s1_dataset


# ============================================================
# Evaluation functions
# ============================================================

def evaluate_full_test(C, test_loader):
    """
    Evaluate Classifier C on the full MNIST test set.
    Returns accuracy as a percentage.
    """
    C.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            preds  = C(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Full MNIST test accuracy : {accuracy:.2f}%")
    return accuracy


def evaluate_s0(C, test_dataset, n=100, seed=42):
    """
    Evaluate Classifier C on S0 —
    100 random real MNIST test images.
    Returns classification error as a percentage.
    """
    torch.manual_seed(seed)
    indices   = torch.randperm(len(test_dataset))[:n]
    s0_imgs   = torch.stack(
        [test_dataset[i][0] for i in indices]
    ).to(DEVICE)
    s0_labels = torch.tensor(
        [test_dataset[i][1] for i in indices]
    ).to(DEVICE)

    C.eval()
    with torch.no_grad():
        preds = C(s0_imgs).argmax(dim=1)

    correct = (preds == s0_labels).sum().item()
    error   = 100 - (100 * correct / n)

    print(f"\nS0 Evaluation (100 real MNIST images)")
    print(f"Correct : {correct}/100")
    print(f"Error   : {error:.2f}%")

    # Show misclassifications
    wrong = (preds != s0_labels).nonzero(as_tuple=True)[0]
    if len(wrong) > 0:
        for idx in wrong:
            print(f"  Image {idx.item():3d}: "
                  f"True={s0_labels[idx].item()}, "
                  f"Predicted={preds[idx].item()}")
    else:
        print("No errors on S0.")

    return error


def evaluate_s1(C, s1_loader):
    """
    Evaluate Classifier C on S1 —
    100 fake GAN generated images.
    Returns classification error as a percentage.
    """
    C.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in s1_loader:
            imgs   = imgs.to(DEVICE)
            preds  = C(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    all_preds  = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    correct = (all_preds == all_labels).sum().item()
    total   = len(all_labels)
    error   = 100 - (100 * correct / total)

    print(f"\nS1 Evaluation (100 fake GAN images)")
    print(f"Correct : {correct}/{total}")
    print(f"Error   : {error:.2f}%")

    # Show misclassifications
    wrong = (all_preds != all_labels).nonzero(as_tuple=True)[0]
    if len(wrong) > 0:
        for idx in wrong:
            print(f"  Image {idx.item():3d}: "
                  f"True={all_labels[idx].item()}, "
                  f"Predicted={all_preds[idx].item()}")
    else:
        print("No errors on S1.")

    return error


def run_full_evaluation(C):
    """
    Run complete evaluation pipeline:
    - Full MNIST test accuracy
    - S0 classification error
    - S1 classification error
    """
    print("=" * 50)
    print("FULL EVALUATION")
    print("=" * 50)

    # Full test set
    test_loader, test_dataset = get_mnist_test_loader()
    full_acc = evaluate_full_test(C, test_loader)

    # S0
    s0_error = evaluate_s0(C, test_dataset)

    # S1
    s1_loader, _ = get_s1_loader()
    s1_error = evaluate_s1(C, s1_loader)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Full MNIST test accuracy : {full_acc:.2f}%")
    print(f"S0 classification error  : {s0_error:.2f}%")
    print(f"S1 classification error  : {s1_error:.2f}%")

    return full_acc, s0_error, s1_error


# ============================================================
# Quick test — runs without trained models
# ============================================================
if __name__ == '__main__':
    print("Testing data loaders...")

    # Test MNIST loader
    test_loader, test_dataset = get_mnist_test_loader()
    print(f"MNIST test samples  : {len(test_dataset)}")

    imgs, labels = next(iter(test_loader))
    print(f"Batch shape         : {imgs.shape}")
    print(f"Label range         : {labels.min()} - {labels.max()}")

    print("\nevaluate.py verified.")