import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
import torchvision
import cv2
from PIL import Image
from einops import rearrange, repeat
import time
import torch.nn.functional as F
import unicodedata
import re
import matplotlib.pyplot as plt

text_path = {
    'train': 'data/train.txt',
    'val': 'data/val.txt',
    'test': 'data/test.txt'
}

generate_type = {
    'iv_s': ['train', 'data/in_vocab.subset.tro.37'],
    'iv_u': ['test',  'data/in_vocab.subset.tro.37'],
    'oov_s': ['train', 'data/oov.common_words'],
    'oov_u': ['test',  'data/test.txt']
}

# ---------------------------------------
# Global Switch for Arabic
# ---------------------------------------
SHOW_HARAKAT = False  # or True if you want diacritics 

# ---------------------------------------
# Arabic / Unifont Setup
# ---------------------------------------
# For an Arabic scenario:
arabic_chars   = "ءأإآابتثجحخدذرزسشصضطظعغفقكلمنهويىئؤة"
arabic_numbers = "٠١٢٣٤٥٦٧٨٩"
english_numbers= "0123456789"
punctuation    = "!\"#%&'()*+-./:<=>@[\\]^_`{|}~،؛؟ " 

# Combined letters (for content mapping)
letters = arabic_chars + arabic_numbers + english_numbers + punctuation

style_len = 416 #target width in pixels for style images (around avg. width of images in the dataset)

# ---------------------------------------
# GLOBAL writer-ID ↔ integer lookup (shared across splits)
# ---------------------------------------
_GLOBAL_WID2IDX = None  # str  -> int
_GLOBAL_IDX2WID = None  # int  -> str


def _build_writer_lookup():
    """Scan all text files once and build a global writer-id mapping."""
    global _GLOBAL_WID2IDX, _GLOBAL_IDX2WID
    if _GLOBAL_WID2IDX is not None:
        return  # already initialised

    writer_set = set()
    for path in text_path.values():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    first_field = line.split(' ', 1)[0]
                    wid = first_field.split(',')[0]
                    writer_set.add(wid)
        except FileNotFoundError:
            # Some splits (e.g. val) may not exist – just skip
            continue

    _GLOBAL_WID2IDX = {w: i for i, w in enumerate(sorted(writer_set))}
    _GLOBAL_IDX2WID = {i: w for w, i in _GLOBAL_WID2IDX.items()}

# ---------------------------------------
# Arabic Helper Functions
# ---------------------------------------

def preprocess_text(text: str) -> str:
    """
    For the new logical-order pipeline we do *not* touch the order of digits.
    You can still normalise whitespace or do other cleaning here if you want.
    """
    return unicodedata.normalize("NFC", text)

def effective_length(text):
    """
    Computes the effective length of a text, ignoring any harakāt (diacritics).
    The text is first normalized to NFD so that diacritics are decomposed.
    Then, any character that is a Unicode combining mark is filtered out.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return len([ch for ch in decomposed if not unicodedata.combining(ch)])

# ---------------------------------------
# Contextual-forms mapping (Arabic subset)
# ---------------------------------------
forms_mapping = {
    'ب': {'isolated': 0x0628, 'initial': 0xFE91, 'medial': 0xFE92, 'final': 0xFE90},
    'ت': {'isolated': 0x062A, 'initial': 0xFE97, 'medial': 0xFE98, 'final': 0xFE96},
    'ث': {'isolated': 0x062B, 'initial': 0xFE9B, 'medial': 0xFE9C, 'final': 0xFE9A},
    'ج': {'isolated': 0x062C, 'initial': 0xFE9F, 'medial': 0xFEA0, 'final': 0xFE9E},
    'ح': {'isolated': 0x062D, 'initial': 0xFEA3, 'medial': 0xFEA4, 'final': 0xFEA2},
    'خ': {'isolated': 0x062E, 'initial': 0xFEA7, 'medial': 0xFEA8, 'final': 0xFEA6},
    'س': {'isolated': 0x0633, 'initial': 0xFEB3, 'medial': 0xFEB4, 'final': 0xFEB2},
    'ش': {'isolated': 0x0634, 'initial': 0xFEB7, 'medial': 0xFEB8, 'final': 0xFEB6},
    'ص': {'isolated': 0x0635, 'initial': 0xFEBB, 'medial': 0xFEBC, 'final': 0xFEBA},
    'ض': {'isolated': 0x0636, 'initial': 0xFEBF, 'medial': 0xFEC0, 'final': 0xFEBE},
    'ط': {'isolated': 0x0637, 'initial': 0xFEC3, 'medial': 0xFEC4, 'final': 0xFEC2},
    'ظ': {'isolated': 0x0638, 'initial': 0xFEC7, 'medial': 0xFEC8, 'final': 0xFEC6},
    'ع': {'isolated': 0x0639, 'initial': 0xFECB, 'medial': 0xFECC, 'final': 0xFECA},
    'غ': {'isolated': 0x063A, 'initial': 0xFECF, 'medial': 0xFED0, 'final': 0xFECE},
    'ف': {'isolated': 0x0641, 'initial': 0xFED3, 'medial': 0xFED4, 'final': 0xFED2},
    'ق': {'isolated': 0x0642, 'initial': 0xFED7, 'medial': 0xFED8, 'final': 0xFED6},
    'ك': {'isolated': 0x0643, 'initial': 0xFEDB, 'medial': 0xFEDC, 'final': 0xFEDA},
    'ل': {'isolated': 0x0644, 'initial': 0xFEDF, 'medial': 0xFEE0, 'final': 0xFEDE},
    'م': {'isolated': 0x0645, 'initial': 0xFEE3, 'medial': 0xFEE4, 'final': 0xFEE2},
    'ن': {'isolated': 0x0646, 'initial': 0xFEE7, 'medial': 0xFEE8, 'final': 0xFEE6},
    'ه': {'isolated': 0x0647, 'initial': 0xFEEB, 'medial': 0xFEEC, 'final': 0xFEEA},
    'و': {'isolated': 0x0648, 'initial': 0x0648, 'medial': 0x0648, 'final': 0xFEEE},
    'ي': {'isolated': 0x064A, 'initial': 0xFEF3, 'medial': 0xFEF4, 'final': 0xFEF2},
    'ا': {'isolated': 0x0627, 'initial': 0x0627, 'medial': 0x0627, 'final': 0xFE8E},
    'د': {'isolated': 0x062F, 'initial': 0x062F, 'medial': 0x062F, 'final': 0xFEAA},
    'ذ': {'isolated': 0x0630, 'initial': 0x0630, 'medial': 0x0630, 'final': 0xFEAC},
    'ر': {'isolated': 0x0631, 'initial': 0x0631, 'medial': 0x0631, 'final': 0xFEAE},
    'ز': {'isolated': 0x0632, 'initial': 0x0632, 'medial': 0x0632, 'final': 0xFEB0},
    'ة': {'isolated': 0x0629, 'initial': 0x0629, 'medial': 0x0629, 'final': 0xFE94},
    'أ': {'isolated': 0x0623, 'initial': 0x0623, 'medial': 0x0623, 'final': 0xFE84},
    'إ': {'isolated': 0x0625, 'initial': 0x0625, 'medial': 0x0625, 'final': 0xFE88},
    'آ': {'isolated': 0x0622, 'initial': 0x0622, 'medial': 0x0622, 'final': 0xFE82},
    'ء': {'isolated': 0x0621, 'initial': 0x0621, 'medial': 0x0621, 'final': 0x0621},
    'ؤ': {'isolated': 0x0624, 'initial': 0x0624, 'medial': 0x0624, 'final': 0xFE86},
    'ى': {'isolated': 0x0649, 'initial': 0x0649, 'medial': 0x0649, 'final': 0xFEF0},
    'ئ': {'isolated': 0x0626, 'initial': 0xFE8B, 'medial': 0xFE8C, 'final': 0xFE8A}
}

non_joining_letters = set("اأإآدذرزوؤىء")

# Helper to decide contextual form key (isolated/initial/medial/final)
def _decide_form(word: str, idx: int) -> str:
    ch = word[idx]
    if ch not in forms_mapping:
        return 'isolated'
    L = len(word)
    prev = word[idx-1] if idx>0 else None
    next_ = word[idx+1] if idx < L-1 else None
    prev_join = prev and prev not in non_joining_letters and prev in forms_mapping
    next_join = next_ and ch not in non_joining_letters and next_ in forms_mapping
    if prev_join and next_join:
        return 'medial'
    elif prev_join and not next_join:
        return 'final'
    elif not prev_join and next_join:
        return 'initial'
    else:
        return 'isolated'

def shape_arabic_text(text: str, letter2index: dict):
    """Light-weight shaper that returns *base-character* indices while also
    computing contextual-form tags for debugging/analysis.

    The returned `indices` are suitable for indexing `con_symbols` which is
    constructed in the same order as `letters` (base characters only).
    `forms` contains the chosen contextual form string for each Arabic letter
    ("isolated", "initial", …) so that downstream visualisers can still paint
    the correct glyph if they wish.
    """

    text = preprocess_text(text)

    indices: list[int] = []
    forms:   list[str] = []

    for idx, ch in enumerate(text):
        if ch in arabic_chars:
            form = _decide_form(text, idx)
            forms.append(form)
        else:
            forms.append('default')

        # Append index (outside the else so both branches hit it)
        if ch in letter2index:
            indices.append(letter2index[ch])
        else:
            # fall back to PAD token index (last row in con_symbols)
            indices.append(letter2index.get('PAD_TOKEN', len(letter2index)))

    return indices, forms

def strip_harakat(text):
    """
    Remove only the Arabic harakāt in `harakat_set`,
    but preserve all other combining marks (e.g. hamza above/below, madda).
    """
    harakat_set = set("ًٌٍَُِّْ")
    
    # 1) Decompose to separate base + combining marks
    decomposed = unicodedata.normalize("NFD", text)
    out_chars = []
    for ch in decomposed:
        # if it's a combining character *and* one of our harakāt, skip it
        if unicodedata.combining(ch) and ch in harakat_set:
            continue
        # otherwise keep it
        out_chars.append(ch)
    # 2) Re-compose so e.g. ALEF + HAMZA_ABOVE → 'أ'
    return unicodedata.normalize("NFC", "".join(out_chars))


def plot_glyphs(glyphs, word, labels):
    """
    Plots glyph images horizontally. The first glyph is actually the last char in word.
    """
    num_chars = glyphs.shape[0]
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars * 2, 2))
    rev_word = word[::-1]  # so label matches the reversed glyph
    for i, ax in enumerate(axes):
        ax.imshow(glyphs[i].numpy(), cmap='gray')
        ax.set_title(f"{rev_word[i]}\n{labels[i]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def split_writer_id(wr_id):
    return tuple(wr_id.split('-',1)) if '-' in wr_id else (wr_id, "")

# =======================================
# IAMDataset for Training/Inferences
# =======================================
class IAMDataset(Dataset):
    def __init__(self, 
                 image_path,
                 style_path,
                 laplace_path,
                 type,
                 content_type='unifont',
                 max_len=10):
        
        self.max_len = max_len
        self.style_len = style_len
        self.split     = type
        
        self.data_dict = self.load_data(text_path[type])
        self.image_path   = os.path.join(image_path,   type)
        self.style_root   = os.path.join(style_path,   type)
        self.laplace_root = os.path.join(laplace_path, type)
        
        # these are used for the content (Arabic)
        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}

        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        
        self.indices = list(self.data_dict.keys())

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)
        self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
                                   ).to(torch.float32).view(1, 1, 3, 3).contiguous()

        # Build / fetch global writer-ID lookup
        _build_writer_lookup()
        self.wid2idx = _GLOBAL_WID2IDX
        self.idx2wid = _GLOBAL_IDX2WID

    def load_data(self, data_path):
        """
        Expects lines like:
           alexuw-648,648-1.jpg شأن
        Then s_id='alexuw-648', image='648-1.jpg', transcription='شأن'
        If SHOW_HARAKAT is False, diacritics are stripped from transcription.
        Lines with transcription length > self.max_len are skipped.
        """
        full_dict = {}
        idx = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            first_field, transcription = parts
            first_parts = first_field.split(',')
            if len(first_parts) < 2:
                continue
            s_id  = first_parts[0]
            image = first_parts[1]

            if not SHOW_HARAKAT:
                transcription = strip_harakat(transcription)

            if effective_length(transcription) > self.max_len:
                continue

            full_dict[idx] = {'s_id': s_id, 'image': image, 'label': transcription}
            idx += 1

        return full_dict

    def get_style_ref(self, wr_id):
        """
        Match English loader approach: keep natural dimensions, pad to max width
        """
        prefix, suffix = split_writer_id(wr_id)
        files = os.listdir(self.style_root)

        # Collect all candidate images for the requested writer
        if prefix == "iesk":            # special naming scheme
            candidates = [f for f in files if f.endswith(f"_{suffix}.bmp")]
        elif prefix == "ahawp":         # another special scheme
            key = f"user{int(suffix):03d}"
            candidates = [f for f in files if f.startswith(key + "_")]
        else:                            # default  "suffix_"  or  "suffix-"
            candidates = [f for f in files
                          if f.startswith(suffix + "_") or f.startswith(suffix + "-")]

        if len(candidates) < 2:
            raise RuntimeError(f"No style images for writer '{wr_id}' in {self.style_root}")

        # Randomly pick anchor & positive
        pick = random.sample(candidates, 2)
        style_images = [cv2.imread(os.path.join(self.style_root, fn), 0) for fn in pick]
        laplace_images = [cv2.imread(os.path.join(self.laplace_root, fn), 0) for fn in pick]

        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [style_image/255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        
        return new_style_images, new_laplace_images
    
    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            syms = pickle.load(f)

        # Store for contextual look-ups in the dataset as well (needed for contextual glyphs)
        self.symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in syms}
        
        # Build content tensor including contextual forms
        contents = []
        for char in self.letters:
            if char in forms_mapping:
                # For Arabic letters, use isolated form as base
                code_point = forms_mapping[char]['isolated']
                symbol = torch.from_numpy(self.symbols[code_point]).float()
            else:
                # For non-Arabic chars, use basic form
                symbol = torch.from_numpy(self.symbols[ord(char)]).float()
            contents.append(symbol)
            
        # Append blank PAD token
        contents.append(torch.zeros_like(contents[0]))
        
        # Stack into single tensor
        contents = torch.stack(contents)
        return contents

    def __len__(self):
        return len(self.indices)
    
    # --------------------------------------------------
    # Borrowed from original IAM loader (GANwriting)
    # Pads a list of character indices with the special PAD token so
    # that all strings in a batch share the same length.
    # --------------------------------------------------
    def label_padding(self, labels, max_len):
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        sample     = self.data_dict[self.indices[idx]]
        img_path   = os.path.join(self.image_path, sample['image'])
        image      = Image.open(img_path).convert('RGB')
        image      = self.transforms(image)

        style_arr, lap_arr = self.get_style_ref(sample['s_id'])
        style   = torch.from_numpy(style_arr).float()
        laplace = torch.from_numpy(lap_arr).float()

        return {
            'img':        image,
            'content':    sample['label'],
            'style':      style,
            'laplace':    laplace,
            'wid':        sample['s_id'],
            'transcr':    sample['label'],
            'image_name': sample['image']
        }


    def collate_fn_(self, batch):
        """
        Collate function that matches English loader's approach while handling
        Arabic-specific features like writer ID mapping and debug prints.
        """
        # Get dimensions (same as English)
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        # Style width handling (same as English)
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        # Initialize tensors (same as English)
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16, 16], dtype=torch.float32)
        
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            # Main image (same as English)
            try:
                imgs[idx, :, :, :item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)

            # ---------- CONTENT (contextual prototypes) ----------
            try:
                # Shape text to get contextual form tag per char
                _, forms = shape_arabic_text(item['content'], self.letter2index)

                glyph_list = []
                for ch, form in zip(item['content'], forms):
                    if ch in forms_mapping:
                        cp = forms_mapping[ch][form]
                    else:
                        cp = ord(ch)
                    mat = self.symbols.get(cp, np.zeros((16,16), dtype=np.float32))
                    glyph_list.append(torch.from_numpy(mat).float())

                glyph_stack = torch.stack(glyph_list)           # [L,16,16]
                content_ref[idx, :len(glyph_stack)] = glyph_stack 
            except Exception as e:
                print(f'content error: {e}', item['content'])

            # Target (same indices as content)
            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])
            
            # Style/Laplace (same as English)
            try:
                cur_w = item['style'].shape[2]
                style_ref[idx, :, :, :cur_w] = item['style']
                laplace_ref[idx, :, :, :cur_w] = item['laplace']
            except Exception as e:
                raise RuntimeError(f"Style/laplace tensor copy failed for sample {idx} | style shape {item['style'].shape} | laplace shape {item['laplace'].shape}") from e

        # Get writer IDs (using global mapping for Arabic)
        wid_tensor = torch.tensor([self.wid2idx[item['wid']] for item in batch], dtype=torch.long)
        wid_str = [item['wid'] for item in batch]  # Keep original strings for reference

        content_ref = 1.0 - content_ref # invert the image

        return {
            'img': imgs,
            'style': style_ref,
            'content': content_ref,
            'wid': wid_tensor,
            'wid_str': wid_str,
            'laplace': laplace_ref,
            'target': target,
            'target_lengths': target_lengths,
            'image_name': image_name,
            'transcr': transcr
        }

"""random sampling of style images during inference"""
class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, laplace_path, ref_num):

        self.style_path   = style_path
        self.laplace_path = laplace_path
        self.ref_num      = ref_num

        # padding parameter (max allowed width when batching)
        self.style_len = style_len  # global constant defined at top of file

        # every file name in style_path corresponds to one writer
        self.author_id = os.listdir(self.style_path)

    def __len__(self):
        return self.ref_num

    def get_style_ref(self, wr_id):
        """
        Directly load the style and laplace image corresponding to the given file name.
        """
        s_path = os.path.join(self.style_path, wr_id)
        l_path = os.path.join(self.laplace_path, wr_id)  # assuming laplace images share the same file name
        s_img = cv2.imread(s_path, flags=0)
        l_img = cv2.imread(l_path, flags=0)
        if s_img is None or l_img is None:
            raise RuntimeError(f"Error reading style or laplace image for file '{wr_id}' in {self.style_path}")

        # Validate pre-processing: expect height exactly 64 px.
        if s_img.shape[0] != 64:
            raise RuntimeError(
                f"Style image '{wr_id}' height is {s_img.shape[0]} (expected 64). "
                "Run the preprocessing script first."
            )

        style_image   = s_img.astype(np.float32) / 255.0
        laplace_image = l_img.astype(np.float32) / 255.0
        return style_image, laplace_image



    def __getitem__(self, _):
        """
        Gather one style/laplace image per file (author) and unify them into a single batch.
        """
        batch = []
        for idx in self.author_id:
            style_img, laplace_img = self.get_style_ref(idx)
            # Convert to tensors with an added channel dimension.
            style_t = torch.from_numpy(style_img).unsqueeze(0).to(torch.float32)    # [1,256,256]
            laplace_t = torch.from_numpy(laplace_img).unsqueeze(0).to(torch.float32)  # [1,256,256]
            batch.append({
                'style': style_t,
                'laplace': laplace_t,
                'wid': idx
            })

        # Unify the batch using similar logic as in IAMDataset.
        s_width = [item['style'].shape[2] for item in batch]
        max_s_width = max(s_width) if max(s_width) < self.style_len else self.style_len

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)

        wid_list = []
        for i, item in enumerate(batch):
            cur_w = item['style'].shape[2]
            style_ref[i, :, :, :cur_w] = item['style']
            laplace_ref[i, :, :, :cur_w] = item['laplace']
            wid_list.append(item['wid'])

        return {
            'style': style_ref,   # [N,1,256,256]
            'laplace': laplace_ref,  # [N,1,256,256]
            'wid': wid_list
        }

# =======================================
# Prepare the Content Image During Inference
# =======================================
class ContentData(IAMDataset):

    def __init__(self, content_type='unifont'):
        # letters used for fallback
        self.letters = letters

        # Build extended letter2index that includes contextual forms
        self.letter2index = {}
        for i, char in enumerate(letters):
            if char in forms_mapping:
                # For Arabic letters, map all forms to consecutive indices
                for form_type in ['isolated', 'initial', 'medial', 'final']:
                    code_point = forms_mapping[char][form_type]
                    self.letter2index[chr(code_point)] = i  # All forms map to base char index
            else:
                # For non-Arabic chars, just map the basic form
                self.letter2index[char] = i
        
        # Load symbols and build content tensor (also stores self.symbols)
        self.con_symbols = self.get_symbols(content_type)

        # Expose shaper for convenience (used by external tests)
        self.shape_arabic_text = shape_arabic_text

    def get_content(self, text):
        """Get content tensor with proper contextual forms for Arabic text."""
        # First get the contextual forms
        _, forms = shape_arabic_text(text, self.letter2index)
        
        # Convert text to proper contextual form characters
        contextual_text = ""
        for char, form in zip(text, forms):
            if char in forms_mapping:
                code_point = forms_mapping[char][form]
                contextual_text += chr(code_point)
            else:
                contextual_text += char
        
        # Build glyphs list using contextual form code points
        glyphs_list = []
        for ch in contextual_text:
            cp = ord(ch)
            mat = self.symbols.get(cp)
            if mat is None:
                # Fallback to blank if glyph missing
                mat = np.zeros((16, 16), dtype=np.float32)
            glyphs_list.append(torch.from_numpy(mat).float())

        content_ref = torch.stack(glyphs_list)
        content_ref = 1.0 - content_ref  # invert to match English pipeline
        return content_ref.unsqueeze(0)

    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            syms = pickle.load(f)

        # Store for contextual look-ups in the dataset as well (needed for contextual glyphs)
        self.symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in syms}
        
        # Build content tensor including contextual forms
        contents = []
        for char in self.letters:
            if char in forms_mapping:
                # For Arabic letters, use isolated form as base
                code_point = forms_mapping[char]['isolated']
                symbol = torch.from_numpy(self.symbols[code_point]).float()
            else:
                # For non-Arabic chars, use basic form
                symbol = torch.from_numpy(self.symbols[ord(char)]).float()
            contents.append(symbol)
            
        # Append blank PAD token
        contents.append(torch.zeros_like(contents[0]))
        
        # Stack into single tensor
        contents = torch.stack(contents)
        return contents
