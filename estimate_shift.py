# estimate_shift.py — Phase 2: BBSE Label Shift Estimation
# BN stats are updated FIRST on static.pt, then BBSE is run on the adapted model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from model_submission import RobustClassifier


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DRIVE_BASE = "/content/drive/My Drive/hackenza-2026-test-time-adaptation-in-the-wild"
WEIGHTS_PATH = "/content/drive/My Drive/weights_v2.pth"
VAL_PATH = os.path.join(DRIVE_BASE, "val_sanity.pt")
TARGET_PATH = os.path.join(DRIVE_BASE, "static.pt")
WT_SAVE_PATH = "/content/drive/My Drive/w_hat_t.npy"

NUM_CLASSES = 10
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Phase 2 on: {DEVICE}")

# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────


def load_labeled(filepath, batch_size=BATCH_SIZE):
    data = torch.load(filepath)
    images = data['images'].float()
    if images.max() > 1.0:
        images = images / 255.0
    labels = data['labels'].long()
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def load_unlabeled(filepath, batch_size=BATCH_SIZE):
    data = torch.load(filepath)
    images = data['images'].float()
    if images.max() > 1.0:
        images = images / 255.0
    return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)

# ──────────────────────────────────────────────
# Step 0 — Load model
# ──────────────────────────────────────────────


def load_model(weights_path):
    model = RobustClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model

# ──────────────────────────────────────────────
# Step 1 — BN Stats Update on target FIRST
# This fixes the softmax collapse before BBSE runs
# ──────────────────────────────────────────────


def update_bn_stats(model, target_loader):
    print("Updating BN stats on target stream...")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for (images,) in target_loader:
            _ = model(images.to(DEVICE))
    model.eval()
    print("BN stats updated.")
    return model

# ──────────────────────────────────────────────
# Step 2 — Soft Confusion Matrix on val_sanity
# ──────────────────────────────────────────────


def compute_confusion_matrix(model, val_loader):
    C = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    with torch.no_grad():
        for images, labels in val_loader:
            probs = F.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy()
            for i, lbl in enumerate(labels.numpy()):
                C[:, lbl] += probs[i]
                class_counts[lbl] += 1
    class_counts[class_counts == 0] = 1
    C = C / class_counts[np.newaxis, :]
    print("Soft confusion matrix computed.")
    return C

# ──────────────────────────────────────────────
# Step 3 — Average softmax over target stream
# ──────────────────────────────────────────────


def average_softmax(model, target_loader):
    mu = np.zeros(NUM_CLASSES, dtype=np.float64)
    n = 0
    with torch.no_grad():
        for (images,) in target_loader:
            probs = F.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy()
            mu += probs.sum(axis=0)
            n += probs.shape[0]
    mu /= n
    print(f"Average softmax over {n} target samples computed.")
    return mu

# ──────────────────────────────────────────────
# Step 4 — BBSE: solve C.T @ w_hat = mu_t
# ──────────────────────────────────────────────


def bbse(C, mu_t):
    lambda_reg = 1e-3
    A = C.T @ C + lambda_reg * np.eye(NUM_CLASSES)
    b = C.T @ mu_t
    w_hat = np.linalg.solve(A, b)
    w_hat = np.clip(w_hat, 0, None)
    w_hat = w_hat / (w_hat.sum() + 1e-12)
    return w_hat


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model(WEIGHTS_PATH)
    target_loader = load_unlabeled(TARGET_PATH)
    val_loader = load_labeled(VAL_PATH)

    # BN update FIRST — fixes softmax collapse
    model = update_bn_stats(model, target_loader)

    # Now run BBSE on the BN-adapted model
    C = compute_confusion_matrix(model, val_loader)
    mu_t = average_softmax(model, target_loader)
    w_hat = bbse(C, mu_t)

    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    print("\n--- Estimated class weights (w_hat_t) ---")
    for cls, w in zip(classes, w_hat):
        print(f"  {cls:<12}: {w:.4f}")

    np.save(WT_SAVE_PATH, w_hat)
    print(f"\nSaved w_hat_t to {WT_SAVE_PATH}")
