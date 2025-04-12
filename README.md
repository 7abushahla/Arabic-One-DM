# One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation
_Hamza A. Abushahla, Ariel Justine Navarro Panopio, Sarah Elfattal, and Imran A. Zualkernan_

This repository contains code and resources for the paper: "[One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234)".

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
To train our model effectively, we required a dataset of Arabic handwritten words that included both ground truth annotations and writer information. Given the scarcity of large-scale, high-quality handwriting datasets in Arabic (in contrast to well-established English datasets such as IAM or CVL), we opted to merge three publicly available datasets: IFN/ENIT, AlexU-Word, and the words portion of the AHAWP (Arabic Handwritten Alphabets, Words, and Paragraphs per User) dataset. This combination increases writer diversity and lexical coverage, providing a more representative sample of Arabic handwriting styles and variations.

### IFN/ENIT Dataset

The IFN/ENIT dataset consists of 26,459 images of handwritten Arabic words representing 937 Tunisian town and village names. These names may comprise one or more Arabic words and occasionally include digits or diacritics, namely the shadda (ـّ), though not all writers include it in their handwriting. Data was collected from 411 different writers, with each word appearing at least three times. All images were cropped and binarized (black ink on a white background), providing clean input for handwriting recognition tasks.

### AlexU-Word Dataset

The AlexU-Word dataset, collected at the Faculty of Engineering, Alexandria University, contains 25,114 word images corresponding to 109 unique Arabic words. These samples were gathered from 907 writers. The selected words were designed to cover all positional variations of Arabic letters (initial, medial, final, and isolated) and were chosen to be short, simple, and free of diacritics. Each writer completed a one-page form consisting of 28 words. Four distinct form templates were used to cover the complete set of letter cases.

Images were originally tightly cropped and binarized (white ink on a black background). For consistency with IFN/ENIT, we inverted all images during preprocessing. We also removed incorrectly cropped images and those containing misspelled or out-of-vocabulary words.

### AHAWP Dataset (Words Portion)

The AHAWP dataset includes handwritten samples of 65 Arabic character forms, 10 Arabic words, and 3 paragraphs, produced by 82 writers. Each participant wrote each word and character 10 times, yielding 8,144 word images. For this work, we utilized only the word portion of the dataset. As with the other datasets, the handwriting samples were collected using structured forms, where each cell was extracted and stored as an individual image.

Because the original AHAWP images were not binarized or tightly cropped, we implemented an automated preprocessing pipeline. The pipeline applied Otsu’s thresholding (with inversion) to binarize each image, detected external contours, computed the union of bounding boxes, added padding, and performed cropping. The resulting images were then binarized once more to ensure uniform formatting.

Additionally, word images with extensive scribbles or strike-throughs were discarded if the writing was obscured. In cases where the core word was still legible, we retained the image after manual correction. These steps ensured consistency across all datasets in terms of image quality and formatting.


### IESK-ArDB Dataset
...

#### Final Compiled Arabic Handwriting Dataset

| Dataset                | Total Words | Unique Words | Writers |
|------------------------|------------|-------------|---------|
| ALEX-U Words           | 25,092     | 109         | 906     |
| IFN/ENIT               | 26,459     | 937         | 411     |
| AHAWP (Words Only)     | 8,137      | 10          | 82      |
| IESK-ArDB              | 2,749      | 364         | 18      |
| **Final Unified Dataset** | **62,437** | **1,412** | **1,414** |



## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```bibtex
@Article{abushahla2025cognitive,
AUTHOR = {Abushahla, Hamza A. and Panopio, Ariel J. N. Elfattal, Sarah and Zualkernan, Imran A.},
TITLE = {One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation},
JOURNAL = { },
VOLUME = {},
YEAR = {},
NUMBER = {},
ARTICLE-NUMBER = {},
URL = {},
ISSN = {},
ABSTRACT = {},
DOI = {}
}
```

You can also reach out through email to: 
- Hamza Abushahla - b00090279@alumni.aus.edu
