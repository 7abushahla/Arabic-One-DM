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

Our methodology builds on the One-DM approach by leveraging GNU Unifont as the foundational source for our glyph representations. Recognizing the context‐sensitive nature of Arabic, we generate each letter’s four typical contextual forms—isolated, initial, medial, and final—by employing a strategy that forces joining using a dummy letter (س). Each glyph is rendered on a 16×16 pixel canvas and subsequently converted into a binary NumPy array.

To further enhance the accuracy of our context-sensitive rendering, we have incorporated a refined joining heuristic that dynamically determines the appropriate contextual form of each Arabic letter based on its neighboring characters. The key aspects of this heuristic are:

- **Non-Joining Letters:**  
  Certain Arabic letters—specifically "ا", "أ", "إ", "آ", "د", "ذ", "ر", "ز", and "و"—do not connect to the following letter. When these letters occur in the middle of a word, they are rendered using their final form if preceded by a joinable letter; otherwise, they appear in their isolated form.

- **Joinable Letters:**  
  For letters that can join on both sides, the heuristic evaluates the neighboring characters as follows:
  - **Medial Form:** A letter is rendered in its medial form if it is both preceded and followed by joinable characters.
  - **Initial Form:** If a letter is not preceded by a joinable character but is followed by one, it is rendered in its initial form.
  - **Final Form:** If a letter is preceded by a joinable character but is not followed by one, it is rendered in its final form.
  - **Isolated Form:** When neither neighboring character is joinable, the letter remains in its isolated form.

Our glyph rendering function further enhances visual consistency by aligning each character to a common baseline using font metrics. This ensures that all characters within a word remain uniformly aligned, preventing issues such as certain letters (e.g., "ـسـ") appearing to float off the line.

### Explanation of Key Components

- **`arabic_reshaper`**  
  This library applies the standard contextual reshaping rules to individual Arabic characters. It converts each letter into its correct form (isolated, initial, medial, or final) based on general joining rules, but it does not account for the dynamic structure of a full word.

- **`bidi.algorithm.get_display`**  
  The BiDi algorithm ensures that the reshaped Arabic text is correctly ordered for right-to-left display. It rearranges the characters so that they render naturally for Arabic reading.

- **`shape_arabic_text` Function**  
  Our custom `shape_arabic_text` function goes beyond the capabilities of `arabic_reshaper` by analyzing the surrounding characters in a word. It:
  - Examines neighboring letters to decide whether a letter should be rendered in its initial, medial, final, or isolated form.
  - Uses a joining heuristic that checks for non-joining letters to decide if a letter joins to the left, right, both, or neither.
  - Reverses the order of glyph indices to ensure the final rendering adheres to Arabic’s right-to-left orientation.

In addition to processing the basic Arabic letters, our pipeline expands to include their contextual variants along with Arabic and English numerals, punctuation, and special symbols. The final output is a pickle file that maps these contextual forms and additional glyphs, ready for integration with the provided One-DM code for accurate, context-aware text rendering.


## Training Datasets

We needed to find a dataset of Arabic handwritten words where the writer information and grounf truth about each handwritten word image are available. the KHATT dataset was inaccessible. For our task, we aquired the following datasets: IFN/ENIT, AlexU Word, and the words portion of the Arabic handwritten alphabets, words and paragraphs per user (AHAWP) dataset... Since Arabic is low-resource in this domain, merging them maximizes handwriting diversity while keeping a large number of writers...Compared to English (IAM, CVL, etc.), Arabic lacks large-scale, high-quality handwriting datasets.....

The datasets are as follows:
- IFN/ENIT dataset: contains 937 Tunisian town/village names, each can be composed by one Arabic word or more. Arabic digits can be present in the town name. also harakat on occasion. The dataset contains a total of 26,459 Arabic word images written by 411 different writers. Each word appears at least 3 times in the database. all images were cropped and binarized (black writing, white background).
- AlexU-Word dataset: Collected at the Faculty of Engineering, Alexandria University, Egypt, contains 25,114 word images from 109 unique Arabic words. The word samples are collected from 907 different writers. The 109 Arabic words in the dataset were chosen to cover all possible cases for each letter in the alphabet. The words were selected to be short and simple, with no diacritics present. Each writer was asked to ﬁll out only a one-page form. Each form contained a table of 28 Arabic words.Four different form models were used to cover all possible cases of Arabic letter. All images were tightky cropped and binarized (white writing, black background), so in preprocessing we inverted the images to match IFN/ENIT. Some images were removed during inspection where words were out of the dataset's vocabulary or misspelded. additionally, some were not cropped tighly so we made sure to crop them properly.





