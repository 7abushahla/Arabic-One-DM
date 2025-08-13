#!/usr/bin/env python3
import torch

# Check English model structure
print("=== English Model Structure ===")
english_model = torch.load("model_zoo/vae_HTR138.pth", map_location='cpu')
print(f"Total parameters: {len(english_model)}")

# Count and show prefixes
vae_keys = [k for k in english_model.keys() if k.startswith('vae.')]
htr_keys = [k for k in english_model.keys() if not k.startswith('vae.')]

print(f"VAE parameters: {len(vae_keys)}")
print(f"HTR parameters: {len(htr_keys)}")

print("\nFirst 5 VAE keys:")
for i, key in enumerate(vae_keys[:5]):
    print(f"  {key}")
    
print("\nLast 5 VAE keys:")
for i, key in enumerate(vae_keys[-5:]):
    print(f"  {key}") 

print("\nFirst 5 HTR keys:")
for i, key in enumerate(htr_keys[:5]):
    print(f"  {key}")

print("\nLast 5 HTR keys:")
for i, key in enumerate(htr_keys[-5:]):
    print(f"  {key}") 
    
    

# Check English model structure
print("=== Arabic Model Structure ===")
arabic_model = torch.load("ocr_checkpoints/ocr_best_state_recognition_vae.pth", map_location='cpu')
print(f"Total parameters: {len(arabic_model)}")

# Count and show prefixes
vae_keys = [k for k in arabic_model.keys() if k.startswith('vae.')]
htr_keys = [k for k in arabic_model.keys() if not k.startswith('vae.')]

print(f"VAE parameters: {len(vae_keys)}")
print(f"HTR parameters: {len(htr_keys)}")

print("\nFirst 5 VAE keys:")
for i, key in enumerate(vae_keys[:5]):
    print(f"  {key}")

print("\nLast 5 VAE keys:")
for i, key in enumerate(vae_keys[-5:]):
    print(f"  {key}") 

print("\nFirst 5 HTR keys:")
for i, key in enumerate(htr_keys[:5]):
    print(f"  {key}")

print("\nLast 5 HTR keys:")
for i, key in enumerate(htr_keys[-5:]):
    print(f"  {key}") 