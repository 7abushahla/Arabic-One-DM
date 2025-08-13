#!/usr/bin/env python3
"""
examine_ocr_models.py

Script to examine and compare the two OCR models:
1. Original English OCR: ./model_zoo/vae_HTR138.pth
2. Your Arabic OCR: ./ocr_checkpoints/ocr_best_state_recognition.pth

This will help us understand the channel mismatch issue.
"""

import torch
import torch.nn as nn
import numpy as np
from models.recognition import HTRNet
from data_loader.loader_ara import letters

def examine_model_architecture(model, model_name):
    """Print detailed architecture of the model"""
    print(f"\n{'='*50}")
    print(f"EXAMINING {model_name}")
    print(f"{'='*50}")
    
    # Print model structure
    print("\nModel Architecture:")
    print(model)
    
    # Get first conv layer info
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            print(f"\nFirst Conv2d layer: {name}")
            print(f"  Input channels: {module.in_channels}")
            print(f"  Output channels: {module.out_channels}")
            print(f"  Kernel size: {module.kernel_size}")
            print(f"  Stride: {module.stride}")
            print(f"  Padding: {module.padding}")
            break
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return first_conv

def load_and_examine_checkpoint(ckpt_path, model_name):
    """Load checkpoint and examine its contents"""
    print(f"\n{'='*50}")
    print(f"LOADING CHECKPOINT: {model_name}")
    print(f"Path: {ckpt_path}")
    print(f"{'='*50}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # If it's a full training checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Found 'model_state_dict' key - this is a training checkpoint")
                if 'epoch' in checkpoint:
                    print(f"Epoch: {checkpoint['epoch']}")
                if 'best_cer' in checkpoint:
                    print(f"Best CER: {checkpoint['best_cer']}")
            else:
                # Direct state dict
                state_dict = checkpoint
                print("Direct state_dict format")
        else:
            print("Unknown checkpoint format")
            return None, None
            
        print(f"\nState dict keys ({len(state_dict)} total):")
        for i, key in enumerate(list(state_dict.keys())[:10]):  # Show first 10 keys
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {i+1:2d}. {key} -> {shape}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more keys")
            
        # Look specifically for the first conv layer
        first_conv_key = None
        for key in state_dict.keys():
            if 'features.0.weight' in key or 'conv1.weight' in key:
                first_conv_key = key
                break
        
        if first_conv_key:
            first_conv_weight = state_dict[first_conv_key]
            print(f"\nFirst conv layer weight: {first_conv_key}")
            print(f"  Shape: {first_conv_weight.shape}")
            print(f"  Expected format: [out_channels, in_channels, kernel_h, kernel_w]")
            if len(first_conv_weight.shape) >= 2:
                print(f"  Input channels: {first_conv_weight.shape[1]}")
                print(f"  Output channels: {first_conv_weight.shape[0]}")
        
        return checkpoint, state_dict
    
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None, None

def test_model_input_compatibility():
    """Test what input shapes each model expects"""
    print(f"\n{'='*50}")
    print("TESTING INPUT COMPATIBILITY")
    print(f"{'='*50}")
    
    # Create test inputs
    batch_size = 2
    height, width = 64, 256
    
    input_3ch = torch.randn(batch_size, 3, height, width)
    input_4ch = torch.randn(batch_size, 4, height, width)
    
    print(f"Test input shapes:")
    print(f"  3-channel: {input_3ch.shape}")
    print(f"  4-channel: {input_4ch.shape}")
    
    # Test Arabic model (yours)
    print(f"\n--- Testing Arabic OCR Model ---")
    n_classes = len(letters) + 1  # +1 for CTC blank
    arabic_model = HTRNet(nclasses=n_classes, vae=True, head='rnn', flattening='maxpool')
    
    try:
        print("Testing 3-channel input...")
        with torch.no_grad():
            output_3ch = arabic_model(input_3ch)
        print(f"✓ 3-channel works: output shape {output_3ch.shape}")
    except Exception as e:
        print(f"✗ 3-channel failed: {e}")
    
    try:
        print("Testing 4-channel input...")
        with torch.no_grad():
            output_4ch = arabic_model(input_4ch)
        print(f"✓ 4-channel works: output shape {output_4ch.shape}")
    except Exception as e:
        print(f"✗ 4-channel failed: {e}")

def main():
    print("OCR Model Examination Script")
    print("="*70)
    
    # Paths to the two models
    english_ocr_path = "./model_zoo/vae_HTR138.pth"
    arabic_ocr_path = "./ocr_checkpoints/ocr_best_state_recognition_vae.pth"
    
    # Check if files exist
    import os
    print(f"\nChecking file existence:")
    print(f"English OCR: {english_ocr_path} -> {'✓ exists' if os.path.exists(english_ocr_path) else '✗ missing'}")
    print(f"Arabic OCR:  {arabic_ocr_path} -> {'✓ exists' if os.path.exists(arabic_ocr_path) else '✗ missing'}")
    
    # Load and examine English OCR (if exists)
    english_ckpt, english_state = None, None
    if os.path.exists(english_ocr_path):
        english_ckpt, english_state = load_and_examine_checkpoint(english_ocr_path, "English OCR")
    
    # Load and examine Arabic OCR
    arabic_ckpt, arabic_state = None, None
    if os.path.exists(arabic_ocr_path):
        arabic_ckpt, arabic_state = load_and_examine_checkpoint(arabic_ocr_path, "Arabic OCR")
    
    # Test input compatibility
    test_model_input_compatibility()
    
    # Compare architectures if both models loaded
    if english_state and arabic_state:
        print(f"\n{'='*50}")
        print("COMPARING ARCHITECTURES")
        print(f"{'='*50}")
        
        english_keys = set(english_state.keys())
        arabic_keys = set(arabic_state.keys())
        
        common_keys = english_keys & arabic_keys
        english_only = english_keys - arabic_keys
        arabic_only = arabic_keys - english_keys
        
        print(f"Common keys: {len(common_keys)}")
        print(f"English-only keys: {len(english_only)}")
        print(f"Arabic-only keys: {len(arabic_only)}")
        
        if english_only:
            print(f"\nKeys only in English model:")
            for key in sorted(english_only):
                print(f"  {key}")
        
        if arabic_only:
            print(f"\nKeys only in Arabic model:")
            for key in sorted(arabic_only):
                print(f"  {key}")
        
        # Compare shapes of common keys
        print(f"\nShape comparison for common keys:")
        for key in sorted(common_keys):
            eng_shape = english_state[key].shape
            ara_shape = arabic_state[key].shape
            match = "✓" if eng_shape == ara_shape else "✗"
            print(f"  {match} {key}: English{eng_shape} vs Arabic{ara_shape}")
    
    print(f"\n{'='*70}")
    print("EXAMINATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 