#!/usr/bin/env python
# coding: utf-8

"""
train_OCR.py

Modified to train a model that matches the English vae_HTR138.pth structure:
- HTRNet for OCR recognition 
- Full VAE component for latent space processing

Example usage:
  python train_OCR.py \
    --image_path data/combined_dataset \
    --style_path data/combined_dataset \
    --laplace_path data/laplace \
    --train_type train \
    --val_type val \
    --test_type test \
    --save_ckpt_dir ./ocr_checkpoints \
    --lr 1e-4 \
    [--resume]
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Avoid 'Bad file descriptor' issue on some Linux systems when many workers are used
mp.set_sharing_strategy('file_system')

# For reshaping Arabic text for proper visualization (if needed)
import arabic_reshaper
from bidi.algorithm import get_display

# For edit distances
import editdistance  # pip install editdistance

import json
import datetime

# 1) Imports from your code:
from data_loader.loader_ara import IAMDataset, letters
from models.recognition import HTRNet

# Import VAE from diffusers
from diffusers import AutoencoderKL

# ----- Define constants -----
BATCH_SIZE = 784     # e.g., 784 or 32 or 64 based on GPU memory
NUM_WORKERS = 64    # e.g., 64 if you have enough CPU cores
MAX_EPOCHS = 1_000     # e.g., up to 2500 in production
EPOCHS_TO_VALIDATE = 50
EPOCHS_TO_CHECKPOINT = 25

PATIENCE = 5

MAX_LEN = 10

# CTC Setup: letters are indexed 0-102, CTC BLANK is at index 103
# The data loader only puts letter indices (0-102) in CTC targets, never PAD tokens
n_classes = len(letters) + 1  # +1 for CTC blank (index 103) 104 classes
BLANK_IDX = len(letters)      # CTC blank is at index 103

# Set random seeds for reproducibility
SEED = 1001
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# def partial_load_onedm(model: nn.Module, checkpoint_path: str):
#     """
#     Partially load 'HTRNet' from a large stable-diffusion style checkpoint,
#     ignoring the 'vae.' submodule and skipping final-layer mismatches.
#
#     model: your HTRNet instance
#     checkpoint_path: path to the .pth file
#     """
#     print(f"Attempting partial load from {checkpoint_path} (skipping 'vae.' and final mismatch).")
#     old_sd = torch.load(checkpoint_path, map_location='cpu')
#
#     # Get the model's current state_dict (so we know shapes).
#     model_sd = model.state_dict()
#
#     filtered_sd = {}
#     for k, v in old_sd.items():
#         # 1) Skip large stable-diffusion VAE submodule keys (which your HTRNet doesn't have).
#         if k.startswith("vae."):
#             continue
#
#         # 2) For the final linear layer, skip if shape mismatches
#         if ("top.fnl.1.weight" in k or "top.fnl.1.bias" in k):
#             if k in model_sd:
#                 if model_sd[k].shape != v.shape:
#                     print(f"Skipping final layer param => {k} due to shape mismatch.")
#                     continue
#             else:
#                 # If doesn't exist in new model
#                 continue
#
#         # 3) Check if the key even exists + shape matches
#         if k in model_sd and model_sd[k].shape == v.shape:
#             filtered_sd[k] = v
#         else:
#             # We skip any unmatched shapes or keys
#             pass
#
#     # Load the filtered checkpoint
#     missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
#     print("Partial load done.")
#     print("Missing keys after partial load:", missing)
#     print("Unexpected keys (ignored):", unexpected)

# --------------------------
# Editdistance-based metrics
# --------------------------
def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate using editdistance on entire Unicode text."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)

def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate, splitting on whitespace."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)

# --------------------------
# Evaluate function (CTC + CER)
# --------------------------
@torch.no_grad()
def evaluate(model, loader, ctc_loss, device, idx_to_char, max_val_batches=None):
    """
    Evaluate the OCR model: compute CTC loss + measure average CER over a (possibly truncated) validation set.
    """
    model.eval()
    total_loss = 0.0
    total_cer  = 0.0
    num_samples= 0
    count      = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_val_batches is not None and batch_idx >= max_val_batches:
            break

        images = batch["img"].to(device)    # No need for ensure_four_channels
        targets       = batch["target"].to(device)
        target_lengths= batch["target_lengths"].to(device)

        # Forward
        logits = model(images)  # => shape [T,B,#classes]
        log_probs = F.log_softmax(logits, dim=2)

        T, B, _ = logits.shape
        input_lengths = torch.IntTensor([T]*B).to(device)

        # CTC loss
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        count += 1

        # Decode each sample for CER
        preds = torch.argmax(log_probs, dim=2)  # shape [T,B]
        for b_idx in range(B):
            raw_seq = preds[:, b_idx].tolist()
            # Basic CTC decode: remove duplicates + blank=0
            dedup = []
            prev = None
            for ch_idx in raw_seq:
                if ch_idx != prev and ch_idx != BLANK_IDX:
                    dedup.append(ch_idx)
                prev = ch_idx
            recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)

            # Ground truth (string)
            gt_str = batch["transcr"][b_idx]

            # CER
            cer_val = compute_cer(gt_str, recognized)
            total_cer += cer_val
            num_samples += 1

        pbar.set_postfix({"val_loss": f"{loss.item():.3f}"})

    avg_loss = (total_loss / count) if count > 0 else 0.0
    avg_cer  = (total_cer / num_samples) if num_samples > 0 else 0.0
    return avg_loss, avg_cer

# @torch.no_grad()
# def evaluate(model, loader, ctc_loss, device, max_val_batches=None):
#     """
#     Evaluate the OCR model using CTC loss over a (possibly truncated) validation set.
#     """
#     model.eval()
#     total_loss = 0.0
#     count = 0

#     pbar = tqdm(loader, desc="Validating", leave=False)
#     for batch_idx, batch in enumerate(pbar):
#         if max_val_batches is not None and batch_idx >= max_val_batches:
#             break

#         images = batch["img"].to(device)
#         targets = batch["target"].to(device)
#         target_lengths = batch["target_lengths"].to(device)

#         logits = model(images)
#         log_probs = F.log_softmax(logits, dim=2)

#         T, B, _ = logits.shape
#         input_lengths = torch.IntTensor([T] * B).to(device)

#         loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
#         total_loss += loss.item()
#         count += 1
#         pbar.set_postfix({"val_loss": f"{loss.item():.3f}"})

#     return total_loss / count if count > 0 else 0.0

# @torch.no_grad()
# def naive_decode_samples(model, loader, device, idx_to_char, max_samples=3):
#     """
#     Decode a few samples for visualization, reshaping Arabic text if needed.
#     """
#     model.eval()
#     data_iter = iter(loader)
#     batch = next(data_iter)

#     images = batch["img"][:max_samples].to(device)
#     ground_truths = batch["transcr"][:max_samples]
#     image_paths = batch["image_name"][:max_samples]

#     logits = model(images)
#     probs  = F.log_softmax(logits, dim=2)
#     preds  = torch.argmax(probs, dim=2).cpu()

#     T, B = preds.shape
#     for i in range(B):
#         raw_seq = preds[:, i].tolist()
#         dedup = []
#         prev = None
#         for ch_idx in raw_seq:
#             if ch_idx != prev and ch_idx != 0:  # Skip blank index 0
#                 dedup.append(ch_idx)
#             prev = ch_idx

#         recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)
#         gt = ground_truths[i]
#         path = image_paths[i]

#         reshaped_gt = get_display(arabic_reshaper.reshape(gt))
#         reshaped_recognized = get_display(arabic_reshaper.reshape(recognized))
#         print(f"Sample {i}: path => '{path}', ground truth => '{reshaped_gt}', recognized => '{reshaped_recognized}'")

@torch.no_grad()
def naive_decode_samples(model, loader, device, idx_to_char, max_samples=3):
    """
    Grabs a small batch and prints the image path, ground truth, and naive decoded result.
    The ground truth and recognized texts are reshaped using arabic_reshaper and bidi for proper display.
    """
    model.eval()
    data_iter = iter(loader)
    batch = next(data_iter)

    images        = batch["img"][:max_samples].to(device)  # No need for ensure_four_channels
    ground_truths = batch["transcr"][:max_samples]  # ground truth transcriptions (as strings)
    image_paths   = batch["image_name"][:max_samples] # paths to the samples

    sample_list = []
    with torch.no_grad():
        logits = model(images)  # expected shape: [T, B, #classes]
        probs  = F.log_softmax(logits, dim=2)
        preds  = torch.argmax(probs, dim=2).cpu()  # shape: [T, B]

    T, B = preds.shape
    for i in range(B):
        raw_seq = preds[:, i].tolist()
        dedup   = []
        prev    = None
        for ch_idx in raw_seq:
            if ch_idx != prev and ch_idx != BLANK_IDX:
                dedup.append(ch_idx)
            prev = ch_idx
        recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)
        gt = ground_truths[i]
        path = image_paths[i]
        
        # Reshape and correct the bidirectional display for Arabic text:
        reshaped_gt = get_display(arabic_reshaper.reshape(gt))
        reshaped_recognized = get_display(arabic_reshaper.reshape(recognized))
        
        print(f"Sample {i}: path => '{path}', ground truth => '{reshaped_gt}', recognized => '{reshaped_recognized}'")
        
        sample_info = {
            "sample_index": i,
            "image_path": path,
            "ground_truth": reshaped_gt,
            "recognized": reshaped_recognized
        }
        sample_list.append(sample_info)
    return sample_list

# ---------------------------------
# Helper: ensure image tensor has 4 channels (duplicate red if necessary)
# ---------------------------------

def ensure_four_channels(img_tensor: torch.Tensor) -> torch.Tensor:
    """If img_tensor has shape [B,3,H,W] duplicate the first channel to obtain [B,4,H,W]."""
    if img_tensor.dim() == 4 and img_tensor.shape[1] == 3:
        return torch.cat([img_tensor, img_tensor[:, 0:1, ...]], dim=1)
    return img_tensor

class OCRModelWithVAE(nn.Module):
    """
    Combined model that includes both HTRNet and VAE to match the English model structure.
    This trains on latent space data like the English version.
    """
    def __init__(self, nclasses, vae_model_path):
        super(OCRModelWithVAE, self).__init__()
        
        # Load pretrained VAE (same as used in main pipeline)
        self.vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae")
        # Freeze VAE parameters - we only train the HTRNet part
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # HTRNet expects 4-channel input (latent space)
        self.htr_net = HTRNet(nclasses=nclasses, vae=True, head='rnn', flattening='maxpool')
        
    def forward(self, images):
        """
        Forward pass:
        1. Encode images to latent space using VAE
        2. Scale latents (like in main pipeline)
        3. Pass through HTRNet for recognition
        """
        # Encode to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            # Scale latents like in the main pipeline
            latents = latents * 0.18215
            
        # HTRNet processes the latent space
        return self.htr_net(latents)
        
    def get_htr_parameters(self):
        """Return only HTRNet parameters for training"""
        return self.htr_net.parameters()

    def state_dict(self):
        """Return combined state dict with proper prefixes to match English model"""
        combined_dict = {}
        
        # Add VAE parameters with 'vae.' prefix
        for key, value in self.vae.state_dict().items():
            combined_dict[f"vae.{key}"] = value
            
        # Add HTRNet parameters directly (no prefix)
        for key, value in self.htr_net.state_dict().items():
            combined_dict[key] = value
            
        return combined_dict
        
    def load_state_dict(self, state_dict, strict=True):
        """Load from combined state dict"""
        vae_dict = {}
        htr_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('vae.'):
                vae_dict[key[4:]] = value  # Remove 'vae.' prefix
            else:
                htr_dict[key] = value
                
        # Load only HTRNet parameters (VAE is frozen)
        missing, unexpected = self.htr_net.load_state_dict(htr_dict, strict=strict)
        return missing, unexpected

def main():
    parser = argparse.ArgumentParser()
    # Data paths and types
    parser.add_argument("--image_path", type=str, default="data/combined_dataset",
                        help="Points to the root images directory.")
    parser.add_argument("--style_path", type=str, default="data/combined_dataset")
    parser.add_argument("--laplace_path", type=str, default="data/combined_dataset_laplace")
    parser.add_argument("--train_type", type=str, default="train")
    parser.add_argument("--val_type",   type=str, default="val")
    parser.add_argument("--test_type",  type=str, default="test")

    # Path to the pretrained One-DM OCR model checkpoint
    parser.add_argument("--save_ckpt_dir", type=str, default="./ocr_checkpoints")
    
    # VAE model path (same as used in main pipeline)
    parser.add_argument("--vae_model_path", type=str, default="model_zoo/stable-diffusion-v1-5",
                        help="Path to the stable diffusion model containing the VAE")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint if available")

    # Intervals for checkpointing and validation
    parser.add_argument("--epochs_to_validate", type=int, default=EPOCHS_TO_VALIDATE)
    parser.add_argument("--epochs_to_checkpoint", type=int, default=EPOCHS_TO_CHECKPOINT)
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")

    args = parser.parse_args()

    os.makedirs(args.save_ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets using the constant MAX_LEN
    train_dataset = IAMDataset(
        image_path=args.image_path,
        style_path=args.style_path,
        laplace_path=args.laplace_path,
        type=args.train_type,
        content_type="unifont",
        max_len=MAX_LEN
    )
    val_dataset = IAMDataset(
        image_path=args.image_path,
        style_path=args.style_path,
        laplace_path=args.laplace_path,
        type=args.val_type,
        max_len=MAX_LEN
    )
    test_dataset = IAMDataset(
        image_path=args.image_path,
        style_path=args.style_path,
        laplace_path=args.laplace_path,
        type=args.test_type,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=train_dataset.collate_fn_, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=val_dataset.collate_fn_, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=test_dataset.collate_fn_, num_workers=NUM_WORKERS)

    # Create the combined OCR model with VAE (like English version)
    # Total classes: 103 letters + 1 CTC_BLANK = 104 classes
    # Letters: indices 0-102, CTC_BLANK: index 103
    model = OCRModelWithVAE(nclasses=n_classes, vae_model_path=args.vae_model_path).to(device)

    # Set checkpoint file paths
    ckpt_last = os.path.join(args.save_ckpt_dir, "checkpoint_last_recognition_vae.pth")
    ckpt_model_state = os.path.join(args.save_ckpt_dir, "checkpoint_last_state_recognition_vae.pth")
    ckpt_best = os.path.join(args.save_ckpt_dir, "ocr_best_recognition_vae.pth")
    ckpt_best_state = os.path.join(args.save_ckpt_dir, "ocr_best_state_recognition_vae.pth")

    # Tracking training progress
    start_epoch = 0
    best_cer = float("inf")
    stagnant_epochs = 0
    patience = args.patience

    # Prepare training components - only train HTRNet parameters
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean")
    optimizer = optim.AdamW(model.get_htr_parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=10**(-1/90000))

    # Optionally resume from checkpoint
    if args.resume and os.path.isfile(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_cer = ckpt["best_cer"]
        stagnant_epochs = ckpt["stagnant_epochs"]
        print(f"Resumed at epoch={start_epoch}, best_cer={best_cer:.4f}")

    # Training loop
    # For decoding: indices 0-102 correspond to actual characters (letters)
    # Index 103 (BLANK_IDX) is reserved for the CTC blank and is **NOT** mapped to a character.
    # PAD tokens (index 103 in data loader) are not used in CTC targets
    idx_to_char = {i: ch for i, ch in enumerate(letters)}
    # --- Informational output ---
    print(f"[CONFIG] n_classes = {n_classes}, BLANK_IDX = {BLANK_IDX}")
    # ---------------------------
    early_stopped = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar):
            # No need for ensure_four_channels - model handles the conversion internally
            images = batch["img"].to(device)
            targets = batch["target"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)
            T, B, _ = logits.shape
            input_lengths = torch.IntTensor([T]*B).to(device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            # Update the tqdm progress bar description
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute average training loss for the epoch
        train_loss = running_loss / batch_count

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f"LR updated from {current_lr:.6e} to {new_lr:.6e}")

        # Save checkpoint every EPOCHS_TO_CHECKPOINT epochs
        if (epoch + 1) % args.epochs_to_checkpoint == 0:
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_cer": best_cer,
                "stagnant_epochs": stagnant_epochs,
            }
            torch.save(model.state_dict(), ckpt_model_state)
            torch.save(ckpt_data, ckpt_last)
            print(f"Checkpoint saved => {ckpt_last}")
        else:
            print(f"Skipping checkpoint saving this epoch (only done every {args.epochs_to_checkpoint} epochs).")

        # Validate every EPOCHS_TO_VALIDATE epochs
        if (epoch + 1) % args.epochs_to_validate == 0:
            val_loss, val_cer = evaluate(model, val_loader, criterion, device, idx_to_char)
            print(f"[Every {EPOCHS_TO_VALIDATE} epoch check] VALIDATION: End of epoch {epoch+1}, val_loss={val_loss:.4f}, CER={val_cer:.4f}")

            # Call our new sample decoding function:
            sample_outputs = naive_decode_samples(model, val_loader, device, idx_to_char, max_samples=5)

            if val_cer < best_cer:
                best_cer = val_cer
                stagnant_epochs = 0
                ckpt_data = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_cer": best_cer,
                    "stagnant_epochs": stagnant_epochs,
                }
                torch.save(model.state_dict(), ckpt_best_state)
                torch.save(ckpt_data, ckpt_best)
                print(f"** New best => val_loss={val_loss:.4f}, CER={best_cer:.4f}, saved best model based on lowest CER.")
            else:
                stagnant_epochs += 1
                print(f"No improvement in CER for {stagnant_epochs} epoch(s).")

                if stagnant_epochs >= patience:
                    print("Early stopping triggered (no CER improvement). Restoring best weights...")
                    model.load_state_dict(torch.load(ckpt_best_state, map_location=device))
                    early_stopped = True

                    # Get the current timestamp
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Compose the log entry with epoch, timestamp, train_loss, and val_loss
                    log_entry = {
                    "epoch": epoch + 1,
                    "timestamp": current_time,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "CER": val_cer,
                    "best_cer": best_cer,
                    "early_stopped": early_stopped,
                    "samples": sample_outputs  # Append the sample decoding results
                    }

                    # Append the new log entry to the JSON file (each entry on a separate line)
                    with open("train_OCR_validation_log_recognition_vae.json", "a") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")

                    break

            # Get the current timestamp
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Compose the log entry with epoch, timestamp, train_loss, and val_loss
            log_entry = {
            "epoch": epoch + 1,
            "timestamp": current_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "CER": val_cer,
            "best_cer": best_cer,
            "early_stopped": early_stopped,
            "samples": sample_outputs  # Append the sample decoding results
            }

            # Append the new log entry to the JSON file (each entry on a separate line)
            with open("train_OCR_validation_log_recognition_vae.json", "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            
        else:
            print(f"Skipping official validation (done every {EPOCHS_TO_VALIDATE} epochs).")

    print(f"Training done. best_cer={best_cer:.4f}")
    
    if early_stopped:
        print("Early stopped. Weights have been restored to best model state.")

    # Final test evaluation
    model.eval()
    test_loss, test_cer = evaluate(model, test_loader, criterion, device, idx_to_char)
    print(f"Test set => CTC loss={test_loss:.4f}, CER={test_cer:.4f}")

    # Also compute WER on the test set:
    test_wer_sum = 0.0
    test_wer_count = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"].to(device)

            logits = model(images)
            probs  = F.log_softmax(logits, dim=2)
            preds  = torch.argmax(probs, dim=2)

            T, B = preds.shape
            for b_idx in range(B):
                raw_seq = preds[:, b_idx].tolist()
                dedup = []
                prev = None
                for ch_idx in raw_seq:
                    if ch_idx != prev and ch_idx != BLANK_IDX:
                        dedup.append(ch_idx)
                    prev = ch_idx
                recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)

                gt_str = batch["transcr"][b_idx]
                # measure WER
                wer_val = compute_wer(gt_str, recognized)
                test_wer_sum += wer_val
                test_wer_count += 1

    test_wer = test_wer_sum / max(1, test_wer_count)
    print(f"Test set => WER={test_wer:.4f}")

    # (Optional) Show a small decode from test
    naive_decode_samples(model, test_loader, device, idx_to_char, max_samples=5)
    print("Done.")

if __name__ == "__main__":
    main()