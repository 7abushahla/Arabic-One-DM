#!/usr/bin/env python3
import torch
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

from finetune_recognition import OCRModelWithVAE, letters, n_classes

def test_model_structure():
    print("=== Testing OCRModelWithVAE Structure ===")
    
    # Create model (without GPU)
    try:
        model = OCRModelWithVAE(
            nclasses=n_classes, 
            vae_model_path="model_zoo/stable-diffusion-v1-5"
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Test state_dict structure
    state_dict = model.state_dict()
    print(f"Total parameters: {len(state_dict)}")
    
    # Count prefixes
    vae_keys = [k for k in state_dict.keys() if k.startswith('vae.')]
    htr_keys = [k for k in state_dict.keys() if not k.startswith('vae.')]
    
    print(f"VAE parameters: {len(vae_keys)}")
    print(f"HTR parameters: {len(htr_keys)}")
    
    # Expected: 248 VAE + 47 HTR = 295 total
    expected_total = 295
    expected_vae = 248
    expected_htr = 47
    
    print(f"Expected: {expected_vae} VAE + {expected_htr} HTR = {expected_total} total")
    
    if len(state_dict) == expected_total:
        print("✅ Total parameter count matches!")
    else:
        print(f"❌ Parameter count mismatch: got {len(state_dict)}, expected {expected_total}")
    
    if len(vae_keys) == expected_vae:
        print("✅ VAE parameter count matches!")
    else:
        print(f"❌ VAE parameter count mismatch: got {len(vae_keys)}, expected {expected_vae}")
    
    if len(htr_keys) == expected_htr:
        print("✅ HTR parameter count matches!")
    else:
        print(f"❌ HTR parameter count mismatch: got {len(htr_keys)}, expected {expected_htr}")
    
    print("\nFirst 5 HTR keys:")
    for i, key in enumerate(htr_keys[:5]):
        print(f"  {key}")
    
    print("\nLast 5 HTR keys:")
    for i, key in enumerate(htr_keys[-5:]):
        print(f"  {key}")
    
    # Test forward pass with dummy data
    try:
        print("\n=== Testing Forward Pass ===")
        # Create dummy RGB image batch
        dummy_images = torch.randn(2, 3, 64, 256)  # Batch=2, 3 channels, H=64, W=256
        
        with torch.no_grad():
            output = model(dummy_images)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            print(f"Expected output shape: [T, B, {n_classes}] where T=sequence length, B=batch size")
            
            if output.shape[1] == 2 and output.shape[2] == n_classes:
                print("✅ Output dimensions look correct!")
            else:
                print(f"❌ Output dimensions unexpected: {output.shape}")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n=== Summary ===")
    if (len(state_dict) == expected_total and 
        len(vae_keys) == expected_vae and 
        len(htr_keys) == expected_htr):
        print("✅ All tests passed! Model structure is correct for training.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = test_model_structure()
    if not success:
        sys.exit(1) 