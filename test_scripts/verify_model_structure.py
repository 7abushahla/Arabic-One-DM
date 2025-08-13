#!/usr/bin/env python3
"""
verify_model_structure.py

Script to verify that our Arabic OCR model will have exactly the same structure 
as the English vae_HTR138.pth model.
"""

import torch
from diffusers import AutoencoderKL
from models.recognition import HTRNet
from data_loader.loader_ara import letters

# Import our new model class
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("finetune_recognition", "finetune_recognition.py")
finetune_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune_module)
OCRModelWithVAE = finetune_module.OCRModelWithVAE

def compare_model_structures():
    print("🔍 VERIFYING MODEL STRUCTURE COMPATIBILITY")
    print("=" * 60)
    
    # 1. Load English model
    print("📖 Loading English model...")
    try:
        english_ckpt = torch.load('./model_zoo/vae_HTR138.pth', map_location='cpu')
        english_keys = set(english_ckpt.keys())
        print(f"   ✅ English model keys: {len(english_keys)}")
    except Exception as e:
        print(f"   ❌ Failed to load English model: {e}")
        return False
    
    # 2. Create our Arabic model
    print("🏗️  Creating Arabic model...")
    try:
        n_classes = len(letters) + 1  # 104 classes
        arabic_model = OCRModelWithVAE(nclasses=n_classes, vae_model_path="model_zoo/stable-diffusion-v1-5")
        arabic_state_dict = arabic_model.state_dict()
        arabic_keys = set(arabic_state_dict.keys())
        print(f"   ✅ Arabic model keys: {len(arabic_keys)}")
    except Exception as e:
        print(f"   ❌ Failed to create Arabic model: {e}")
        return False
    
    # 3. Compare structures
    print("🔄 Comparing structures...")
    
    # Check key counts
    if len(english_keys) != len(arabic_keys):
        print(f"   ❌ Key count mismatch: English={len(english_keys)}, Arabic={len(arabic_keys)}")
        return False
    else:
        print(f"   ✅ Key count matches: {len(english_keys)}")
    
    # Check key names (excluding final layer which may have different output size)
    english_structure_keys = {k for k in english_keys if not k.endswith(('fnl.1.weight', 'fnl.1.bias'))}
    arabic_structure_keys = {k for k in arabic_keys if not k.endswith(('fnl.1.weight', 'fnl.1.bias'))}
    
    if english_structure_keys != arabic_structure_keys:
        missing_in_arabic = english_structure_keys - arabic_structure_keys
        extra_in_arabic = arabic_structure_keys - english_structure_keys
        print(f"   ❌ Structure mismatch!")
        if missing_in_arabic:
            print(f"      Missing in Arabic: {list(missing_in_arabic)[:5]}...")
        if extra_in_arabic:
            print(f"      Extra in Arabic: {list(extra_in_arabic)[:5]}...")
        return False
    else:
        print(f"   ✅ Model structure matches (excluding final layer)")
    
    # 4. Check specific layer shapes (excluding final layer)
    print("📏 Checking layer shapes...")
    shape_mismatches = []
    
    for key in english_structure_keys:
        if key in arabic_state_dict:
            eng_shape = english_ckpt[key].shape
            ara_shape = arabic_state_dict[key].shape
            if eng_shape != ara_shape:
                shape_mismatches.append((key, eng_shape, ara_shape))
    
    if shape_mismatches:
        print(f"   ❌ Shape mismatches found:")
        for key, eng_shape, ara_shape in shape_mismatches[:5]:
            print(f"      {key}: English={eng_shape}, Arabic={ara_shape}")
        return False
    else:
        print(f"   ✅ All layer shapes match")
    
    # 5. Check final layer dimensions
    print("🎯 Checking final layer...")
    eng_final_weight = english_ckpt['top.fnl.1.weight']
    ara_final_weight = arabic_state_dict['top.fnl.1.weight']
    
    print(f"   English final layer: {eng_final_weight.shape} (input={eng_final_weight.shape[1]}, output={eng_final_weight.shape[0]})")
    print(f"   Arabic final layer: {ara_final_weight.shape} (input={ara_final_weight.shape[1]}, output={ara_final_weight.shape[0]})")
    
    if eng_final_weight.shape[1] != ara_final_weight.shape[1]:
        print(f"   ❌ Input dimension mismatch!")
        return False
    else:
        print(f"   ✅ Input dimensions match: {eng_final_weight.shape[1]}")
        print(f"   ℹ️  Output classes: English={eng_final_weight.shape[0]}, Arabic={ara_final_weight.shape[0]} (expected difference)")
    
    print("\n🎉 MODEL STRUCTURE VERIFICATION COMPLETE!")
    print("✅ Your Arabic model will have EXACTLY the same structure as the English model")
    print("✅ The only difference is the number of output classes (as expected)")
    print("✅ Ready for training!")
    
    return True

if __name__ == "__main__":
    success = compare_model_structures()
    if not success:
        print("\n❌ Verification failed. Please check the issues above.")
        sys.exit(1)
    else:
        print("\n✅ Verification successful!") 