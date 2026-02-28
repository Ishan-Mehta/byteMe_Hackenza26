# adapt.py — Phase 3: Two-Step Test-Time Adaptation (BN Stats + SAR)
# Run AFTER estimate_shift.py. Requires weights.pth, w_hat_t.npy, and target stream .pt file.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import copy
from model_submission import RobustClassifier

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────


DRIVE_BASE = "/content/drive/My Drive/hackenza-2026-test-time-adaptation-in-the-wild"
WEIGHTS_PATH = "/content/drive/My Drive/weights_v2.pth"
TARGET_PATH = os.path.join(
    DRIVE_BASE, "static.pt")   # <-- update if needed
WT_PATH = "/content/drive/My Drive/w_hat_t.npy"
ADAPTED_SAVE = "/content/drive/My Drive/weights_adapted.pth"

NUM_CLASSES = 10
BATCH_SIZE = 64
SAR_LR = 1e-4
SAR_EPOCHS = 3                          # passes over target stream
# ~0.92  — only adapt uncertain samples
ENTROPY_THRESH = 0.4 * np.log(NUM_CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Phase 3 on: {DEVICE}")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


# def load_unlabeled(filepath, batch_size=BATCH_SIZE):
#     data = torch.load(filepath)
#     images = data['images'].float() / 255.0
#     ds = TensorDataset(images)
#     return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

def load_unlabeled(filepath, batch_size=BATCH_SIZE):
    data = torch.load(filepath)
    images = data['images'].float()
    if images.max() > 1.0:
        images = images / 255.0
    ds = TensorDataset(images)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_model(path):
    model = RobustClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def entropy(probs):
    """Shannon entropy per sample.  probs: [B, C]"""
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1)   # [B]

# ──────────────────────────────────────────────
# Step A — BatchNorm Statistics Update
# Simply forward the entire target stream in train() mode so BN
# running_mean / running_var adapt to the target distribution.
# No gradients, no weight updates.
# ──────────────────────────────────────────────


def update_bn_stats(model, target_loader):
    print("\n[Phase 3 - Step A] Updating BatchNorm statistics...")
    # Reset all BN running stats to force re-estimation
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            # cumulative moving average (more stable)
            m.momentum = None

    model.train()
    with torch.no_grad():
        for (images,) in target_loader:
            images = images.to(DEVICE)
            # BN layers update running stats automatically
            _ = model(images)

    model.eval()
    print("  BatchNorm stats updated.")
    return model

# ──────────────────────────────────────────────
# Step B — SAR: Sharpness-Aware Reliable Entropy Minimisation
# Adjust logits by log(w_hat_t) for label-shift correction,
# then minimise entropy only on uncertain (unreliable) samples.
# ──────────────────────────────────────────────


def sar_adaptation(model, target_loader, w_hat):
    print("\n[Phase 3 - Step B] SAR adaptation...")

    log_w = torch.tensor(np.log(w_hat + 1e-8),
                         dtype=torch.float32).to(DEVICE)  # [C]

    # Only adapt BN + final FC layers — keep backbone frozen for stability
    adapt_params = []
    for name, param in model.named_parameters():
        if 'bn' in name.lower() or 'fc' in name.lower() or 'norm' in name.lower():
            param.requires_grad_(True)
            adapt_params.append(param)
        else:
            param.requires_grad_(False)

    optimizer = torch.optim.SGD(
        adapt_params, lr=SAR_LR, momentum=0.0, weight_decay=0)

    for epoch in range(SAR_EPOCHS):
        total_loss = 0.0
        n_adapted = 0

        for (images,) in target_loader:
            images = images.to(DEVICE)

            model.train()              # keep BN in train mode for stat adaptation
            logits = model(images)                        # [B, C]
            # label-shift correction
            logits_adjusted = logits + log_w.unsqueeze(0)
            probs = F.softmax(logits_adjusted, dim=1)   # [B, C]
            ent = entropy(probs)                       # [B]

            # Reliable filter: only uncertain samples
            uncertain_mask = ent > ENTROPY_THRESH
            if uncertain_mask.sum() == 0:
                continue

            loss = ent[uncertain_mask].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_adapted += uncertain_mask.sum().item()

        print(f"  SAR Epoch [{epoch+1}/{SAR_EPOCHS}] | "
              f"Entropy loss: {total_loss/max(len(target_loader), 1):.4f} | "
              f"Samples adapted: {n_adapted}")

    model.eval()
    return model


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    w_hat = np.load(WT_PATH)
    print(f"Loaded w_hat_t: {np.round(w_hat, 4)}")

    target_loader = load_unlabeled(TARGET_PATH)
    model = load_model(WEIGHTS_PATH)

    # Step A
    model = update_bn_stats(model, target_loader)

    # Step B
    model = sar_adaptation(model, target_loader, w_hat)

    # Save adapted weights
    torch.save(model.state_dict(), ADAPTED_SAVE)
    print(f"\nSaved adapted weights to {ADAPTED_SAVE}")
