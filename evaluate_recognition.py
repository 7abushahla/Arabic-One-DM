#!/usr/bin/env python
# coding: utf-8

"""
evaluate_recognition.py

This script loads a pretrained HTRNet checkpoint (as produced by finetune_recognition.py)
and runs evaluation on the specified dataset split (default: test).

Example usage:
  python evaluate_recognition.py \
      --checkpoint_path ./ocr_checkpoints/checkpoint_last_state_recognition.pth \
      --image_path data/combined_dataset \
      --style_path data/combined_dataset \
      --laplace_path data/combined_dataset_laplace \
      --split test \
      --batch_size 784

Notes:
• The checkpoint should be the pure model.state_dict() file (e.g. *_state_recognition.pth).
  If you pass the "checkpoint_last_recognition.pth" meta-dict, the script will try to
  extract the "model_state_dict" entry automatically.
• The script prints:
    – CTC loss on the full split
    – CER and WER
    – A few naive decoded samples for qualitative inspection.
"""

import os
import argparse
import json
import datetime
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# For Arabic text reshaping during sample printing
import arabic_reshaper
from bidi.algorithm import get_display

import editdistance  # pip install editdistance

# -----------------------------------------------------------------------------
# Project-specific imports
# -----------------------------------------------------------------------------
from data_loader.loader_ara import IAMDataset, letters
from models.recognition import HTRNet

# -----------------------------------------------------------------------------
# Constants & helpers (mirrored from finetune_recognition.py)
# -----------------------------------------------------------------------------
BLANK_IDX = len(letters)  # CTC blank index (after the last character)
N_CLASSES = len(letters) + 1  # characters + blank


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)


def ensure_four_channels(img_tensor: torch.Tensor) -> torch.Tensor:
    """Duplicate the first (red) channel if tensor has only 3 channels."""
    if img_tensor.dim() == 4 and img_tensor.shape[1] == 3:
        return torch.cat([img_tensor, img_tensor[:, 0:1, ...]], dim=1)
    return img_tensor


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, criterion: nn.CTCLoss,
                   device: torch.device, idx_to_char: Dict[int, str]) -> Dict[str, float]:
    """Compute CTC loss, CER, WER over an entire loader split."""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    num_batches = 0
    num_samples = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        images = ensure_four_channels(batch["img"].to(device))
        targets = batch["target"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        logits = model(images)  # [T,B,C]
        log_probs = F.log_softmax(logits, dim=2)
        T, B, _ = logits.shape
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.int32, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(log_probs, dim=2)  # [T,B]
        for b in range(B):
            raw_seq = preds[:, b].tolist()
            dedup = []
            prev = None
            for ch_idx in raw_seq:
                if ch_idx != prev and ch_idx != BLANK_IDX:
                    dedup.append(ch_idx)
                prev = ch_idx
            recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)
            gt_str = batch["transcr"][b]

            total_cer += compute_cer(gt_str, recognized)
            total_wer += compute_wer(gt_str, recognized)
            num_samples += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_cer = total_cer / max(1, num_samples)
    avg_wer = total_wer / max(1, num_samples)
    return {"ctc_loss": avg_loss, "cer": avg_cer, "wer": avg_wer}


@torch.no_grad()
def print_sample_decodes(model: nn.Module, loader: DataLoader, device: torch.device,
                         idx_to_char: Dict[int, str], max_samples: int = 5) -> List[Dict]:
    """Decode a small batch and print the results for qualitative inspection."""
    model.eval()
    batch = next(iter(loader))

    images = ensure_four_channels(batch["img"][:max_samples].to(device))
    gts = batch["transcr"][:max_samples]
    paths = batch["image_name"][:max_samples]

    logits = model(images)
    probs = F.log_softmax(logits, dim=2)
    preds = torch.argmax(probs, dim=2).cpu()

    T, B = preds.shape
    outputs = []
    for i in range(B):
        raw_seq = preds[:, i].tolist()
        dedup = []
        prev = None
        for ch_idx in raw_seq:
            if ch_idx != prev and ch_idx != BLANK_IDX:
                dedup.append(ch_idx)
            prev = ch_idx
        recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)

        reshaped_gt = get_display(arabic_reshaper.reshape(gts[i]))
        reshaped_rec = get_display(arabic_reshaper.reshape(recognized))
        print(f"Sample {i}: path='{paths[i]}', GT='{reshaped_gt}', REC='{reshaped_rec}'")

        outputs.append({
            "index": i,
            "image_path": paths[i],
            "ground_truth": reshaped_gt,
            "recognized": reshaped_rec
        })
    return outputs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained HTRNet OCR checkpoint.")

    # Data arguments
    parser.add_argument("--image_path", type=str, default="data/combined_dataset")
    parser.add_argument("--style_path", type=str, default="data/combined_dataset")
    parser.add_argument("--laplace_path", type=str, default="data/combined_dataset_laplace")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on.")

    # Checkpoint & model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pth checkpoint (state_dict or meta-dict).")

    # Eval hyperparams
    parser.add_argument("--batch_size", type=int, default=784)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=10,
                        help="Max sequence length used when building IAMDataset (should match training).")
    parser.add_argument("--samples_to_print", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    dataset = IAMDataset(
        image_path=args.image_path,
        style_path=args.style_path,
        laplace_path=args.laplace_path,
        type=args.split,
        max_len=args.max_len,
    )
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate_fn_)

    # Build model & load weights
    model = HTRNet(nclasses=N_CLASSES, vae=True, head="rnn", flattening="maxpool").to(device)

    ckpt_path = args.checkpoint_path
    print(f"Loading checkpoint from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # assume pure state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Criterion
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean")

    # Evaluation
    idx_to_char = {i: ch for i, ch in enumerate(letters)}

    metrics = evaluate_split(model, loader, criterion, device, idx_to_char)
    print("\n=== Evaluation results ===")
    print("Split:", args.split)
    print(f"CTC loss: {metrics['ctc_loss']:.4f}")
    print(f"CER:      {metrics['cer']:.4f}")
    print(f"WER:      {metrics['wer']:.4f}")

    print("\nShowing sample decodes …")
    _ = print_sample_decodes(model, loader, device, idx_to_char, max_samples=args.samples_to_print)

    # Optionally log JSON for later aggregations
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "checkpoint": os.path.basename(ckpt_path),
        **metrics,
    }
    json_path = "evaluate_recognition_log.json"
    with open(json_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged results to {json_path}")


if __name__ == "__main__":
    # Ensure reproducibility (same as training script)
    SEED = 1001
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    main() 