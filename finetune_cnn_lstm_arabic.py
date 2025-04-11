#!/usr/bin/env python
# coding: utf-8

"""
finetune_cnn_lstm_arabic.py

Example usage:
  python finetune_cnn_lstm_arabic.py \
    --cfg IAM64_finetune.yml \
    --image_path data/combined_dataset \
    --style_path data/combined_dataset \
    --laplace_path data/laplace \
    --train_type train \
    --val_type val \
    --test_type test \
    --pretrained_cnnlstm models/hw.pt \
    --save_ckpt_dir ./ocr_checkpoints \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    [--resume]
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import random
import numpy as np
from tqdm import tqdm

import arabic_reshaper
from bidi.algorithm import get_display

# For edit distances
import editdistance  # pip install editdistance 

import json
import datetime

# 1) Import your `IAMDataset` from loader_ara.py
from data_loader.loader_ara import IAMDataset, letters  # also contentData if you want
from models.cnn_lstm import create_model  # your CRNN model builder

import unicodedata

# Set random seeds for reproducibility
SEED = 1001 # To follow the same splits during Arabic One-DM training/ train_finetuning

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############### PARTIAL LOAD ###############
def partial_load_cnn_only(model, checkpoint_path):
    """
    Loads only the CNN layers from a checkpoint, skipping final conv or LSTM weights
    so you can partially initialize your CRNN from a different model.
    """
    print(f"Attempt partial load from {checkpoint_path} (CNN only).")
    old_sd = torch.load(checkpoint_path, map_location='cpu')
    filtered = {}
    for k, v in old_sd.items():
        # skip the final conv or LSTM
        if "cnn.conv6" in k or "rnn." in k:
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Partial load done.")
    print("Missing keys after partial load:", missing)
    print("Unexpected keys:", unexpected)
    
    
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
    Evaluate the CRNN model in CTC mode with CER calculation.
    Computes average CTC loss and average CER over a (possibly truncated) validation set.
    """
    model.eval()
    total_loss = 0.0
    total_cer  = 0.0
    num_samples = 0
    count = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_val_batches is not None and batch_idx >= max_val_batches:
            break

        images = batch["img"].to(device)
        targets = batch["target"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # Forward pass: get logits and compute log probabilities.
        logits = model(images)   # Expected shape: [T, B, #classes]
        log_probs = F.log_softmax(logits, dim=2)
        T, B, _ = logits.shape

        # Create an input length tensor (all sequences are assumed to have length T)
        input_lengths = torch.IntTensor([T] * B).to(device)

        # Compute CTC loss for the current batch.
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        count += 1

        # Decode predictions to compute CER.
        preds = torch.argmax(log_probs, dim=2)  # Shape: [T, B]
        for b_idx in range(B):
            raw_seq = preds[:, b_idx].tolist()
            # Basic CTC decoding: remove duplicates and ignore blanks (assumed to be token 0).
            dedup = []
            prev = None
            for ch_idx in raw_seq:
                if ch_idx != prev and ch_idx != 0:
                    dedup.append(ch_idx)
                prev = ch_idx
            
            # Convert token indices to string using idx_to_char mapping.
            recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)

            # Ground truth transcription for the sample.
            gt_str = batch["transcr"][b_idx]

            # Compute the CER using your `compute_cer` function.
            cer_val = compute_cer(gt_str, recognized)
            total_cer += cer_val
            num_samples += 1

        pbar.set_postfix({"val_loss": f"{loss.item():.3f}"})

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    return avg_loss, avg_cer

import torch
import torch.nn.functional as F
import arabic_reshaper
from bidi.algorithm import get_display

@torch.no_grad()
def naive_decode_samples(model, loader, device, idx_to_char, max_samples=3):
    """
    Grabs a small batch and:
      - Prints each sample with proper RTL shaping in the console.
      - Returns a list of dicts with the raw strings for logging.
    """
    model.eval()
    data_iter = iter(loader)
    batch = next(data_iter)

    images        = batch["img"][:max_samples].to(device)
    ground_truths = batch["transcr"][:max_samples]
    image_paths   = batch["image_name"][:max_samples]

    # Forward
    logits    = model(images)                         # [T, B, #classes]
    log_probs = F.log_softmax(logits, dim=2)
    preds     = torch.argmax(log_probs, dim=2).cpu()  # [T, B]

    T, B = preds.shape
    sample_list = []

    for i in range(B):
        # CTC‑style decode
        raw_seq = preds[:, i].tolist()
        dedup   = []
        prev    = None
        for ch_idx in raw_seq:
            if ch_idx != prev and ch_idx != 0:
                dedup.append(ch_idx)
            prev = ch_idx
        recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)

        gt   = ground_truths[i]
        path = image_paths[i]

        # Build visual strings for printing
        vis_gt  = get_display(arabic_reshaper.reshape(gt))
        vis_rec = get_display(arabic_reshaper.reshape(recognized))

        # Print in RTL visually correct form
        print(f"Sample {i}: path => '{path}', "
              f"ground truth => '{vis_gt}', recognized => '{vis_rec}'")

        # Store raw strings for JSON/logging
        sample_list.append({
            "sample_index": i,
            "image_path":   path,
            "ground_truth": gt,
            "recognized":   recognized
        })

    return sample_list
        
############### MAIN ###############
def main():
    BATCH_SIZE = 784 # 1024 # 32 or 64 or 96
    NUM_WORKERS = 64

    MAX_EPOCHS =  1_000
    EPOCHS_TO_VALIDATE =  50 # 100 
    EPOCHS_TO_CHECKPOINT = 25 #  50

    PATIENCE = 5
    
    MAX_LEN = 10

    parser= argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="data/combined_dataset",
                        help="Points to the root images directory.")
    parser.add_argument("--style_path", type=str, default="data/combined_dataset")
    parser.add_argument("--laplace_path", type=str, default="data/combined_dataset_laplace")
    parser.add_argument("--train_type", type=str, default="train")
    parser.add_argument("--val_type",   type=str, default="val")
    parser.add_argument("--test_type",  type=str, default="test")

    parser.add_argument("--pretrained_cnnlstm", type=str, default="models/hw.pt")
    parser.add_argument("--save_ckpt_dir", type=str, default="./ocr_checkpoints")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr",         type=float, default=1e-4)

    
    parser.add_argument("--resume",     action="store_true", help="Resume from checkpoint_last.pth if found")

    args= parser.parse_args()

    os.makedirs(args.save_ckpt_dir, exist_ok=True)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build the IAMDataset for train/val/test
    train_dataset = IAMDataset(
        image_path   = args.image_path,
        style_path   = args.style_path,
        laplace_path = args.laplace_path,
        type         = args.train_type,  # e.g. 'train'
        content_type = 'unifont_arabic', # or whichever
        max_len      = MAX_LEN                # or 10, etc.
    )
    val_dataset = IAMDataset(
        image_path   = args.image_path,
        style_path   = args.style_path,
        laplace_path = args.laplace_path,
        type         = args.val_type,
        max_len      = MAX_LEN
    )
    test_dataset= IAMDataset(
        image_path   = args.image_path,
        style_path   = args.style_path,
        laplace_path = args.laplace_path,
        type         = args.test_type,
        max_len      = MAX_LEN
    )

    train_loader= torch.utils.data.DataLoader(
        train_dataset,
        batch_size= args.batch_size,
        shuffle=True,
        collate_fn= train_dataset.collate_fn_,  # use the custom collate from loader_ara
        num_workers=NUM_WORKERS
    )
    val_loader= torch.utils.data.DataLoader(
        val_dataset,
        batch_size= args.batch_size,
        shuffle=False,
        collate_fn= val_dataset.collate_fn_,
        num_workers=NUM_WORKERS
    )
    test_loader= torch.utils.data.DataLoader(
        test_dataset,
        batch_size= args.batch_size,
        shuffle=False,
        collate_fn= test_dataset.collate_fn_,
        num_workers=NUM_WORKERS
    )

    # 2) Build CRNN
    config= {
       "cnn_out_size": 1024,
       "num_of_channels": 3,
       "num_of_outputs": len(letters) + 1,  # or figure out your ctc blank
       "use_instance_norm": False,
       "nh": 512
    }
    model = create_model(config).to(device)

    # 3) Partial load if old checkpoint
    if os.path.isfile(args.pretrained_cnnlstm):
        partial_load_cnn_only(model, args.pretrained_cnnlstm)
    else:
        print("No old checkpoint => from scratch.")

    # single checkpoint path
    ckpt_last = os.path.join(args.save_ckpt_dir, "checkpoint_last.pth")
    ckpt_model_state = os.path.join(args.save_ckpt_dir, "checkpoint_last_state.pth")

    ckpt_best = os.path.join(args.save_ckpt_dir, "ocr_best.pth")
    ckpt_best_state = os.path.join(args.save_ckpt_dir, "ocr_best_state.pth")

    start_epoch = 0
    best_cer = float("inf")
    stagnant_epochs = 0
    patience = PATIENCE #30                    # Early stopping patience


    # 4) Prepare optimizer, ctc, maybe a scheduler
    criterion= nn.CTCLoss(blank=0, reduction='mean')
    optimizer= optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler (Exponential Decay: 10^(-1/90000))
    scheduler = ExponentialLR(optimizer, gamma=10**(-1/90_000))

    # 5) If resume => load single checkpoint
    if args.resume and os.path.isfile(ckpt_last):
        print(f"Resuming from {ckpt_last}")
        ckpt= torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        stagnant_epochs = ckpt["stagnant_epochs"]
        # best_val_loss= ckpt["best_val_loss"]
        best_cer= ckpt["best_cer"]
        start_epoch= ckpt["epoch"]
        # print(f"Resumed at epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")
        print(f"Resumed at epoch={start_epoch}, best_cer={best_cer:.4f}")
    elif os.path.isfile(args.pretrained_cnnlstm):
        pass  # we only partial-loaded CNN weights
    else:
        print("No resume => starting fresh")

    # 6) Train

    # Wrap your train_loader with tqdm. "leave=True" keeps the bar after completion
    # "desc" is the label shown on the left side of the bar

    # For decoding
    idx_to_char = {i+1: ch for i, ch in enumerate(letters)} 
    # blank=0, so letters map from 1..N

    early_stopped = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss=0.0
        batch_count = 0

        # Use tqdm to show a progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        # for step, batch in enumerate(train_loader):
        for step, batch in enumerate(pbar):
            images= batch["img"].to(device)
            targets=batch["target"].to(device)
            target_lengths= batch["target_lengths"]

            # forward
            logits= model(images)
            log_probs= F.log_softmax(logits, dim=2)
            # Suppose shape is [T,B,#classes], we do:
            T, B, _= logits.shape
            input_lengths = torch.IntTensor([T]*B)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+= loss.item()
            batch_count += 1

            # Update the tqdm progress bar description
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute average training loss for the epoch
        train_loss = running_loss / batch_count
        
        # LR scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f"LR updated from {current_lr:.6e} to {new_lr:.6e}")

        # Save checkpoint every EPOCHS_TO_CHECKPOINT epochs
        if (epoch + 1) % EPOCHS_TO_CHECKPOINT == 0:
            ckpt_data = {
               "epoch": epoch+1,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(),
               "best_cer": best_cer,
            #    "best_val_loss": best_val_loss,
               "stagnant_epochs": stagnant_epochs,
            }
            torch.save(model.state_dict(), ckpt_model_state)
            torch.save(ckpt_data, ckpt_last)
            print(f"Checkpoint saved => {ckpt_last}")
        else:
            print(f"Skipping checkpoint saving (done every {EPOCHS_TO_CHECKPOINT} epochs).")
            
        # Validate every EPOCHS_TO_VALIDATE epochs
        if (epoch + 1) % EPOCHS_TO_VALIDATE == 0:
            val_loss, val_cer = evaluate(model, val_loader, criterion, device, idx_to_char)
            # val_loss = evaluate(model, val_loader, criterion, device, max_val_batches=None)
            print(f"[Every {EPOCHS_TO_VALIDATE} epoch check] VALIDATION: End of epoch {epoch+1}, val_loss={val_loss:.4f}, CER={val_cer:.4f}")
            
            
            # Call our new sample decoding function:
            # naive_decode_samples(model, loader, device, idx_to_char, max_samples=3):
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
                # print(f"** New best => val_loss improved to {best_val_loss:.4f}, saved best model.")
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
                    "samples": sample_outputs  # Append the decoded samples
                    }

                    # Append the new log entry to the JSON file (each entry on a separate line)
                    with open("finetune_validation_log_Muharaf_HW.json", "a") as log_file:
                        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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
            with open("finetune_validation_log_Muharaf_HW.json", "a") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n") 
            
        else:
            print(f"Skipping official validation (done every {EPOCHS_TO_VALIDATE} epochs).")
        
    print(f"Training done. best_cer={best_cer:.4f}")
    # print(f"Training done. best_val_loss={best_val_loss:.4f}")

    if early_stopped:
        print("Early stopped. Weights have been restored to best model state.")
    
    # Final test evaluation
    model.eval()
    # test_loss = evaluate(model, test_loader, criterion, device)
    # print(f"Test set CTC loss => {test_loss:.4f}")
    
    
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
                    if ch_idx != prev and ch_idx != 0:
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
    
    
    
 