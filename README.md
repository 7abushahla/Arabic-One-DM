# One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation

## Introduction

## $Khat^2$ Dataset
Inspired by the $Font^2$ dataset[^1][^2] (which was used to train the ResNet-18 backbones in One-DM[^3] and VATr[^4]) we built a large synthetic dataset of Arabic word images rendered in a wide variety of fonts—including those that mimic handwriting. In our pipeline, over **2,000** freely available Arabic calligraphic fonts were scraped from multiple websites, then manually verified to ensure they correctly render every Arabic character (with decorative fonts containing elements like hearts or stars discarded). Additionally, drawing inspiration from HATFORMER[^5][^6], we collected more than **130** paper background images and compiled an Arabic corpus of **8.2 million** words from diverse online sources, including Wikipedia.

For a chosen number *N* (e.g., 1,000), we randomly select *N* words from the corpus (of varying lengths) and *N* fonts that pass our automated validation checks. We then render every possible combination of the selected words and fonts—resulting in $N×N$ (e.g., 1,000×1,000 = 1,000,000) image samples, with each font representing a distinct class (i.e., 1,000 images per font).

During the synthetic data generation process, for every *(word, font)* pair, the system randomly selects one of the background images and applies random augmentations (such as distortions and blur) before exporting the final image. Ground-truth labels—mapped to the corresponding font names—are stored in a CSV file, and these images are later used to train a ResNet-18 convolutional neural network backbone as part of a style encoder.

- Arabic corpus is from HATFormer (words.pickle): link
- Collected arabic fonts: link
- main generation code, basic fonts, corpus, and backgrounds are borrowed from HATFormer[^6]. we edited them to generate single words instead of lines and to maek the lable the name of the font and to generate the $N×N$ samples as done in $Font^2$. we also edit the augmentaions to be simialr what is used in $Font^2$. 
- we follow their training settings [^1] and train our ResNet-18 model to classify the 1,000,000 samples into the 1,000 font classes to learn the style representations needed to later learn arabic handwriting properly once integrated and made to train on real handwritten images.

[^1]: https://arxiv.org/abs/2304.01842
[^2]: https://github.com/aimagelab/font_square
[^3]: https://arxiv.org/abs/2409.04004
[^4]: https://arxiv.org/abs/2303.15269
[^5]: https://arxiv.org/abs/2410.02179
[^6]: https://zenodo.org/records/14165756

## Arabic GNU Unifont Glyph Mapping
Our methodology builds on the One-DM approach by leveraging GNU Unifont as the foundational source for our glyph representations. Recognizing the context‐sensitive nature of Arabic, we generate each letter’s four typical contextual forms—isolated, initial, medial, and final—by employing a strategy that forces joining using a dummy letter (س). The reshaping of Arabic characters is managed by the `arabic_reshaper` library, which applies the necessary transformation rules, while `bidi.algorithm.get_display` ensures the proper right-to-left orientation. Each glyph is rendered on a 16×16 pixel canvas and subsequently converted into a binary NumPy array, preserving the original pixelated quality of GNU Unifont.

To further enhance the accuracy of our context-sensitive rendering, we have incorporated a refined joining heuristic that dynamically determines the appropriate contextual form of each Arabic letter based on its neighboring characters. The key aspects of this heuristic are:

- **Non-Joining Letters:**  
  Certain Arabic letters—specifically "ا", "أ", "إ", "آ", "د", "ذ", "ر", "ز", and "و"—do not connect to the following letter. When these letters occur in the middle of a word, they are rendered using their final form if preceded by a joinable letter; otherwise, they appear in their isolated form.

- **Joinable Letters:**  
  For letters that can join on both sides, the heuristic evaluates the neighboring characters as follows:
  - **Medial Form:** A letter is rendered in its medial form if it is both preceded and followed by joinable characters.
  - **Initial Form:** If a letter is not preceded by a joinable character but is followed by one, it is rendered in its initial form.
  - **Final Form:** If a letter is preceded by a joinable character but is not followed by one, it is rendered in its final form.
  - **Isolated Form:** When neither neighboring character is joinable, the letter remains in its isolated form.

Additionally, to ensure visual consistency across words, our glyph rendering function aligns each character to a common baseline by leveraging font metrics. This alignment prevents issues such as certain letters (e.g., "ـسـ") appearing to float off the line, thereby ensuring that all characters within a word remain uniformly aligned.

Finally, our text rendering pipeline reverses the order of the glyph indices to accurately simulate right-to-left writing, thus producing contextually correct and visually coherent Arabic text.

In addition to processing the basic Arabic letters, our pipeline expands to include their contextual variants along with Arabic and English numerals, punctuation, and special symbols. The final output is a pickle file that maps these contextual forms and additional glyphs, ready for integration with the provided One-DM code for accurate, context-aware text rendering.






