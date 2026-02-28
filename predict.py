# predict.py — Final Inference → submission.csv
# Generates predictions for both static.pt (Public LB) and test_suite_public.pt (Private LB)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from model_submission import RobustClassifier


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DRIVE_BASE = "/content/drive/My Drive/hackenza-2026-test-time-adaptation-in-the-wild"
WEIGHTS_PATH = "/content/drive/My Drive/weights_v2.pth"
WT_PATH = "/content/drive/My Drive/w_hat_t.npy"
STATIC_PATH = os.path.join(DRIVE_BASE, "static.pt")
SUITE_PATH = os.path.join(DRIVE_BASE, "test_suite_public.pt")
OUTPUT_CSV = "/content/drive/My Drive/submission.csv"

NUM_CLASSES = 10
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {DEVICE}")

# ──────────────────────────────────────────────
# Load model + BN adapt to static.pt first
# ──────────────────────────────────────────────


def load_and_adapt_model(weights_path, static_images):
    model = RobustClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # BN stats update on static.pt (same as estimate_shift.py)
    print("Updating BN stats...")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None

    model.train()
    with torch.no_grad():
        # Forward in batches
        for i in range(0, len(static_images), BATCH_SIZE):
            batch = static_images[i:i+BATCH_SIZE].to(DEVICE)
            _ = model(batch)

    model.eval()
    print("BN stats updated.")
    return model

# ──────────────────────────────────────────────
# Predict with label-shift correction
# ──────────────────────────────────────────────


def predict_batch(model, images, log_w):
    """images: [B, 1, 28, 28] tensor already on CPU, normalized"""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            logits = model(batch)
            # label-shift correction
            logits_adjusted = logits + log_w.unsqueeze(0)
            preds = logits_adjusted.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
    return all_preds

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────


def generate_submission():
    w_hat = np.load(WT_PATH)
    log_w = torch.tensor(np.log(w_hat + 1e-8), dtype=torch.float32).to(DEVICE)
    print(f"Loaded w_hat_t: {np.round(w_hat, 4)}")

    # Load static.pt
    static_data = torch.load(STATIC_PATH)
    static_images = static_data['images'].float()
    if static_images.max() > 1.0:
        static_images = static_images / 255.0

    # Load and adapt model
    model = load_and_adapt_model(WEIGHTS_PATH, static_images)

    results = []

    # 1. Static set (Public LB)
    print("\nPredicting static.pt...")
    preds = predict_batch(model, static_images, log_w)
    for i, p in enumerate(preds):
        results.append({'ID': f'static_{i}', 'Category': p})
    print(f"  {len(preds)} predictions done.")

    # 2. 24-scenario suite (Private LB)
    print("\nPredicting test_suite_public.pt...")
    suite = torch.load(SUITE_PATH)
    scenario_keys = sorted(
        [k for k in suite.keys() if k.startswith('scenario')])

    for skey in scenario_keys:
        images = suite[skey].float()
        if images.max() > 1.0:
            images = images / 255.0
        preds = predict_batch(model, images, log_w)
        for i, p in enumerate(preds):
            results.append({'ID': f'{skey}_{i}', 'Category': p})
        print(f"  {skey}: {len(preds)} predictions done.")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} total predictions to {OUTPUT_CSV}")
    print(
        f"\nLabel distribution:\n{df['Category'].value_counts().sort_index()}")


if __name__ == "__main__":
    generate_submission()
