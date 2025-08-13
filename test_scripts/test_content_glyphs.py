import torch
import matplotlib.pyplot as plt
from data_loader.loader_ara import ContentData
import numpy as np

def test_content_glyphs():
    """Test if Arabic content glyphs are generated correctly"""
    print("Testing Arabic content glyph generation...")
    
    # Initialize ContentData
    content_loader = ContentData(content_type='unifont_arabic')
    
    # Test words
    test_words = ["أستاذ", "مرحبا", "شكرا", "سلام", "وداعا"]
    
    for word in test_words:
        print(f"\n--- Testing word: '{word}' ---")
        
        # Get content glyphs
        glyphs = content_loader.get_content(word)
        print(f"Glyph tensor shape: {glyphs.shape}")
        print(f"Expected length: {len(word)}, Actual: {glyphs.shape[1]}")
        
        # Check if glyphs contain actual content
        non_zero_count = torch.sum(glyphs > 0.5).item()
        total_pixels = glyphs.numel()
        ratio = non_zero_count / total_pixels
        print(f"Non-zero pixels: {non_zero_count}/{total_pixels} ({ratio:.3f})")
        
        # Visualize first few glyphs
        fig, axes = plt.subplots(1, min(len(word), 5), figsize=(2*min(len(word), 5), 2))
        if len(word) == 1:
            axes = [axes]
        
        for i in range(min(len(word), 5)):
            glyph = glyphs[0, i].numpy()  # Remove batch dimension
            axes[i].imshow(glyph, cmap='gray')
            axes[i].set_title(f"'{word[i]}'")
            axes[i].axis('off')
        
        plt.suptitle(f"Glyphs for '{word}'")
        plt.tight_layout()
        plt.savefig(f"glyph_test_{word}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: glyph_test_{word}.png")
        
        # Check if all glyphs are different (not all zeros)
        unique_glyphs = 0
        for i in range(glyphs.shape[1]):
            if torch.sum(glyphs[0, i] > 0.5) > 0:
                unique_glyphs += 1
        print(f"Non-empty glyphs: {unique_glyphs}/{glyphs.shape[1]}")

if __name__ == "__main__":
    test_content_glyphs() 