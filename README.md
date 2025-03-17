# One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation

## $Khat^2$ Dataset
Inspired by the **Font²** dataset, we built a large synthetic dataset of Arabic word images rendered in a wide variety of fonts—including those that mimic handwriting. In our pipeline, over **2,000** freely available Arabic calligraphic fonts were scraped from multiple websites, then manually verified to ensure they correctly render every Arabic character (with decorative fonts containing elements like hearts or stars discarded). Additionally, drawing inspiration from **HATFORMER**, we collected more than **130** paper background images and compiled an Arabic corpus of **8.2 million** words from diverse online sources, including Wikipedia.

For a chosen number *N* (e.g., 1,000), we randomly select *N* words from the corpus (of varying lengths) and *N* fonts that pass our automated validation checks. We then render every possible combination of the selected words and fonts—resulting in *N×N* (e.g., 1,000×1,000 = 1,000,000) image samples, with each font representing a distinct class (i.e., 1,000 images per font).

During the synthetic data generation process, for every *(word, font)* pair, the system randomly selects one of the background images and applies random augmentations (such as distortions and blur) before exporting the final image. Ground-truth labels—mapped to the corresponding font names—are stored in a CSV file, and these images are later used to train a ResNet-18 convolutional neural network backbone as part of a style encoder.

Arabic corpus is from HATFormer (words.pickle): link
Collected arabic fonts: link
