import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F
import unicodedata
import re
import matplotlib.pyplot as plt

# ---------------------------------------
# Global Switch for Arabic
# ---------------------------------------
SHOW_HARAKAT = False  # or True if you want diacritics 

# ---------------------------------------
# Arabic / Unifont Setup
# ---------------------------------------
# For an Arabic scenario:
arabic_chars   = "ءاأإآابتثجحخدذرزسشصضطظعغفقكلمنهويىئؤة"
arabic_numbers = "٠١٢٣٤٥٦٧٨٩"
english_numbers= "0123456789"
punctuation    = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟"
symbols        = "$€£¥¢©®±×÷ "

# Combined letters (for content mapping)
letters = arabic_chars + arabic_numbers + english_numbers + punctuation + symbols

style_len = 416 #target width in pixels for style images (around avg. width of images in the dataset)

# =======================================
# Dataset File Paths and Generation Types
# =======================================
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
# Arabic Helper Functions
# ---------------------------------------

def preprocess_text(text):
    """
    Pre-process the text so that any contiguous sequence of English digits
    is reversed. This ensures that after the overall reversal for RTL display,
    the English numbers appear in their original order.
    """
    def reverse_digits(match):
        return match.group(0)[::-1]
    return re.sub(r'\d+', reverse_digits, text)

def effective_length(text):
    """
    Computes the effective length of a text, ignoring any harakāt (diacritics).
    The text is first normalized to NFD so that diacritics are decomposed.
    Then, any character that is a Unicode combining mark is filtered out.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return len([ch for ch in decomposed if not unicodedata.combining(ch)])

def shape_arabic_text(text, letter2index):
    """
    Processes the input text as follows:
      1. Pre-processes English digit sequences
      2. Normalizes the entire text to NFC
      3. Tokenizes the text into (char, diacritic) pairs for Arabic letters,
         keeping non-Arabic characters unchanged
      4. If SHOW_HARAKAT is False, forces each Arabic letter's diacritic to "base"
      5. Determines the appropriate contextual form (isolated, initial, medial, final)
      6. Returns two lists: glyph indices + forms, reversed for RTL display.
    """
    text = preprocess_text(text)
    text = unicodedata.normalize("NFC", text)
    
    non_joining = set("اأإآدذرزو")
    harakat_set = set("ًٌٍَُِّْ")
    
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in arabic_chars:
            diacritic = "base"
            if i + 1 < n and text[i+1] in harakat_set:
                diacritic = unicodedata.normalize("NFC", text[i+1])
                i += 2
            else:
                i += 1
            tokens.append((char, diacritic))
        elif text[i] in harakat_set:
            i += 1
        else:
            tokens.append((text[i], None))
            i += 1
    
    # If SHOW_HARAKAT = False, forcibly set diacritic to "base"
    if not SHOW_HARAKAT:
        tokens = [(ch, "base") if ch in arabic_chars else (ch, d) for (ch, d) in tokens]
    
    indices = []
    forms_detected = []
    num_tokens = len(tokens)

    for ii, (char, d) in enumerate(tokens):
        if char in arabic_chars:
            prev_join = (ii > 0 and tokens[ii-1][0] in arabic_chars and tokens[ii-1][0] not in non_joining)
            next_join = (ii < num_tokens - 1 and tokens[ii+1][0] in arabic_chars)
            curr_joinable = (char not in non_joining)
            
            if not curr_joinable:
                form = "final" if prev_join else "isolated"
            else:
                if prev_join and next_join:
                    form = "medial"
                elif prev_join and not next_join:
                    form = "final"
                elif not prev_join and next_join:
                    form = "initial"
                else:
                    form = "isolated"

            try:
                idx = letter2index[char][form][d]
            except KeyError:
                idx = letter2index[char]["isolated"]["base"]
            indices.append(idx)
            forms_detected.append(form)
        else:
            # Non-Arabic
            if char in letter2index:
                if isinstance(letter2index[char], dict):
                    idx = letter2index[char].get("default", 0)
                else:
                    idx = letter2index[char]
                form = "default"
            else:
                idx = letter2index["PAD"]
                form = "PAD"
            indices.append(idx)
            forms_detected.append(form)
    
    # Reverse for RTL
    return list(reversed(indices)), list(reversed(forms_detected))

# def strip_harakat(text):
#     """
#     Removes all diacritics (harakāt) from the input text.
#     This function first normalizes the text to NFD (decomposed form) and then
#     filters out any combining characters.
#     """
#     decomposed = unicodedata.normalize("NFD", text)
#     stripped = "".join([ch for ch in decomposed if not unicodedata.combining(ch)])
#     return stripped


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
                 content_type='unifont_arabic',
                 max_len=10):
        
        self.max_len = max_len
        self.style_len = style_len
        self.split     = type
        
        # read lines from e.g. data/train.txt
        data_file = text_path[type]
        self.data_dict = self.load_data(data_file)

        # Now join the 'type' folder to each of the paths.
        # flat combined_dataset with split
        self.image_path   = os.path.join(image_path,   type)
        self.style_root   = os.path.join(style_path,   type)
        self.laplace_root = os.path.join(laplace_path, type)
        
        # these are used for the content (Arabic)
        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        # self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.letter2index = {label: n+1 for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())

        self.transforms = torchvision.transforms.Compose([
            # Force the line image to be 256×256
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])

        # load unifont_arabic or unifont, etc.
        self.con_symbols, _ = self.get_symbols(content_type)

        # example placeholder for a Laplace filter
        self.laplace = torch.tensor([[0, 1, 0],
                                     [1,-4, 1],
                                     [0, 1, 0]], dtype=torch.float32
                                   ).to(torch.float32).view(1, 1, 3, 3).contiguous()

    def get_symbols(self, input_type):
        """
        Load Arabic symbols from data/{input_type}.pickle
        Returns con_symbols + letter2index
        """
        with open(f"data/{input_type}.pickle", "rb") as f:
            data = pickle.load(f)
        glyph_entries = data['with_harakat']['glyph_entries']
        letter2index = data['with_harakat']['letter2index']

        max_idx = max(e['idx'][0] for e in glyph_entries)
        glyph_list = [None]*(max_idx+1)

        for entry in glyph_entries:
            idx_val = entry['idx'][0]
            mat_16x16 = entry['mat'].astype(np.float32)
            glyph_list[idx_val] = torch.from_numpy(mat_16x16)
        con_symbols = torch.stack(glyph_list)

        return con_symbols, letter2index

    
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
        Force 256×256 resizing of style & laplace images so that
        your subsequent code doesn't produce weird shapes in the ResNet.
        """
        prefix, suffix = split_writer_id(wr_id)
        files = os.listdir(self.style_root)

        # special cases
        if prefix == "iesk":
            candidates = [f for f in files if f.endswith(f"_{suffix}.bmp")]
        elif prefix == "ahawp":
            key = f"user{int(suffix):03d}"
            candidates = [f for f in files if f.startswith(key + "_")]
        else:
            candidates = [f for f in files
                          if f.startswith(suffix + "_") or f.startswith(suffix + "-")]

        if len(candidates) < 2:
            raise RuntimeError(f"No style images for writer '{wr_id}' in {self.style_root}")

        pick = random.sample(candidates, 2)
        imgs = [cv2.imread(os.path.join(self.style_root, fn), 0) for fn in pick]
        laps = [cv2.imread(os.path.join(self.laplace_root, fn), 0) for fn in pick]

        # force 256×256
        style_arr = np.zeros((2,256,256), dtype=np.float32)
        lap_arr   = np.zeros((2,256,256), dtype=np.float32)
        for j,(im,lp) in enumerate(zip(imgs,laps)):
            if im is None or lp is None:
                raise RuntimeError(f"Error reading {pick[j]}")
            im2 = cv2.resize(im,  (256,256), interpolation=cv2.INTER_AREA)
            lp2 = cv2.resize(lp, (256,256), interpolation=cv2.INTER_AREA)
            style_arr[j] = im2.astype(np.float32)/255.0
            lap_arr[j]   = lp2.astype(np.float32)/255.0

        return style_arr, lap_arr

    def __len__(self):
        return len(self.indices)

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
        Collate function that pads:
          - main images => same H/W
          - style images => same H, clamp W
          - glyph => used for content
        """
        B = len(batch)

        # ============ 1) MAIN IMAGES (3,H,W) => pad =============
        img_heights = [item['img'].shape[1] for item in batch]
        img_widths  = [item['img'].shape[2] for item in batch]
        max_h = max(img_heights)
        max_w = max(img_widths)

        imgs = torch.ones([B, 3, max_h, max_w], dtype=torch.float32)
        for i, item in enumerate(batch):
            cur_h = item['img'].shape[1]
            cur_w = item['img'].shape[2]
            imgs[i, :, :cur_h, :cur_w] = item['img']

        # ============ 2) STYLE IMAGES (2,256,256) => pad if needed =============
        #   but we already forced them to be 256×256 above, so no mismatch expected
        style_heights = [item['style'].shape[1] for item in batch]  # each is 256
        style_widths  = [item['style'].shape[2] for item in batch]  # each is 256
        max_style_h = max(style_heights)  # probably 256
        raw_max_style_w = max(style_widths)  # also 256
        max_style_w = min(raw_max_style_w, self.style_len)  # e.g. 256 vs style_len=416 => 256

        style_ref   = torch.ones([B, 2, max_style_h, max_style_w], dtype=torch.float32)
        laplace_ref = torch.zeros([B, 2, max_style_h, max_style_w], dtype=torch.float32)

        for i, item in enumerate(batch):
            sh = item['style'].shape[1]  # should be 256
            sw = item['style'].shape[2]  # should be 256
            # clamp if you want (256 vs style_len=416 => 256)
            clamped_w = min(sw, max_style_w)
            style_ref[i, :, :sh, :clamped_w]   = item['style'][:, :sh, :clamped_w]
            laplace_ref[i, :, :sh, :clamped_w] = item['laplace'][:, :sh, :clamped_w]

        # ============ 3) CONTENT => glyph references =============
        transcr = [item['transcr'] for item in batch]
        c_width = [len(txt) for txt in transcr]  # number of chars in each label
        max_c_width = max(c_width)
        content_ref = torch.zeros([B, max_c_width, 16, 16], dtype=torch.float32)
        # also build OCR ctc target
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        max_tlen = max(target_lengths)
        target = torch.zeros([B, max_tlen], dtype=torch.int32)

        for i, item in enumerate(batch):
            label = item['content']
            # For glyph references
            content_inds = [self.letter2index[ch] for ch in label]
            glyphs = self.con_symbols[content_inds]  # [len(label), 16, 16]
            content_ref[i, :len(glyphs)] = glyphs
            # For ctc target
            tinds = [self.letter2index[ch] for ch in label]
            target[i, :len(tinds)] = torch.tensor(tinds, dtype=torch.int32)

        writer_ids  = [item['wid'] for item in batch]
        image_names = [item['image_name'] for item in batch]

        # invert glyph bitmaps if needed
        content_ref = 1.0 - content_ref

        return {
            'img':            imgs,
            'style':          style_ref,
            'laplace':        laplace_ref,
            'content':        content_ref,
            'wid':            writer_ids,
            'transcr':        transcr,
            'target':         target,
            'target_lengths': target_lengths,
            'image_name':     image_names
        }

# ---------------------------------------
# Random_StyleIAMDataset
# ---------------------------------------
class Random_StyleIAMDataset(IAMDataset):
    """
    Now we also forcibly resize the single style image to (256,256)
    so that it won't cause shape mismatches in the fusion code.
    """
    def __init__(self, style_path, laplace_path, ref_num) -> None:
        self.style_path = style_path        # flat folder with all style images
        self.laplace_path = laplace_path    # flat folder with all laplace images
        self.style_len = style_len
        self.ref_num = ref_num

        # self.author_id = self._get_all_writer_ids(text_file) # GPT
        self.author_id = os.listdir(os.path.join(self.style_path))

    def get_style_ref(self, wr_id):
        """
        Directly load the style and laplace image corresponding to the given file name.
        The image is resized to 256x256 if its width is greater than 20.
        """
        import cv2
        s_path = os.path.join(self.style_path, wr_id)
        l_path = os.path.join(self.laplace_path, wr_id)  # assuming laplace images share the same file name
        s_img = cv2.imread(s_path, flags=0)
        l_img = cv2.imread(l_path, flags=0)
        if s_img is None or l_img is None:
            raise RuntimeError(f"Error reading style or laplace image for file '{wr_id}' in {self.style_path}")
        if s_img.shape[1] > 58:
            # Resize both images to 256x256
            s_img = cv2.resize(s_img, (256, 256), interpolation=cv2.INTER_AREA)
            l_img = cv2.resize(l_img, (256, 256), interpolation=cv2.INTER_AREA)
            style_image = s_img.astype(np.float32) / 255.0
            laplace_image = l_img.astype(np.float32) / 255.0
            return style_image, laplace_image
        else:
            raise RuntimeError(f"Style image '{wr_id}' width <= 58 in {self.style_path}")

    def __len__(self):
        return self.ref_num

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
            if max_s_width < self.style_len:
                style_ref[i, :, :, :cur_w] = item['style']
                laplace_ref[i, :, :, :cur_w] = item['laplace']
            else:
                style_ref[i, :, :, :cur_w] = item['style'][:, :, :self.style_len]
                laplace_ref[i, :, :, :cur_w] = item['laplace'][:, :, :self.style_len]
            wid_list.append(item['wid'])

        return {
            'style': style_ref,   # [N,1,256,256]
            'laplace': laplace_ref,  # [N,1,256,256]
            'wid': wid_list
        }

    # def _get_all_writer_ids(self, text_file):
    #         wr_set = set()
    #         with open(text_file, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
    #         for line in lines:
    #             parts = line.strip().split(' ', 1)
    #             if len(parts) < 2:
    #                 continue
    #             first_field = parts[0]  # e.g., "alexuw-648,648-1.jpg"
    #             ff = first_field.split(',', 1)
    #             if len(ff) < 2:
    #                 continue
    #             wr_id = ff[0]
    #             wr_set.add(wr_id)
    #         return list(wr_set)

    # def get_style_ref(self, wr_id):
    #     """
    #     We forcibly resize each style/laplace image to 256×256,
    #     just like in IAMDataset. For writer IDs with prefix "iesk",
    #     we match files ending with _{suffix}.bmp; for all others,
    #     we match files starting with "{suffix}_" or "{suffix}-".
    #     """
    #     import cv2
    #     prefix, suffix = split_writer_id(wr_id)
    #     files = os.listdir(self.style_path)
                
    #     if prefix == "iesk":
    #         cands = [f for f in files if f.endswith(f"_{suffix}.bmp")]
    #     elif prefix == "ahawp":
    #         key = f"user{int(suffix):03d}"
    #         cands = [f for f in files if f.startswith(key + "_")]
    #     else:
    #         cands = [f for f in files if f.startswith(suffix + "_") or f.startswith(suffix + "-")]

    #     random.shuffle(cands)

    #     style_image = None
    #     laplace_image = None

    #     for fn in cands:
    #         s_path = os.path.join(self.style_path, fn)
    #         l_path = os.path.join(self.laplace_path, fn)
    #         s_img = cv2.imread(s_path, flags=0)
    #         l_img = cv2.imread(l_path, flags=0)
    #         if s_img is None or l_img is None:
    #             continue
    #         if s_img.shape[1] > 20:
    #         # if s_img.shape[1] > 58: # GPT
    #             # Now forcibly resize to 256×256
    #             s_img = cv2.resize(s_img, (256, 256), interpolation=cv2.INTER_AREA)
    #             l_img = cv2.resize(l_img, (256, 256), interpolation=cv2.INTER_AREA)
    #             style_image = s_img.astype(np.float32)/255.0
    #             laplace_image = l_img.astype(np.float32)/255.0
    #             break

    #     if style_image is None or laplace_image is None:
    #         raise RuntimeError(f"No style image with width> 20 found or read error for writer '{wr_id}' in {self.style_path}")

    #     return style_image, laplace_image
    
    # def __len__(self):
    #     return self.ref_num

    # def __getitem__(self, _):
    #     """
    #     We'll gather exactly 1 style-laplace image per author,
    #     forcibly sized 256×256, then unify them in a single big batch.
    #     """
    #     batch = []
    #     for idx in self.author_id:
    #         style_img, laplace_img = self.get_style_ref(idx)
    #         # shape is now 256×256
    #         style_t   = torch.from_numpy(style_img).unsqueeze(0).to(torch.float32)    # [1,256,256]
    #         laplace_t = torch.from_numpy(laplace_img).unsqueeze(0).to(torch.float32)  # [1,256,256]
    #         batch.append({
    #             'style':   style_t,
    #             'laplace': laplace_t,
    #             'wid':     idx
    #         })

    #     # unify them (the collate logic is simpler here)
    #     s_width = [item['style'].shape[2] for item in batch]  # each 256
    #     if max(s_width) < self.style_len:
    #         max_s_width = max(s_width)  # 256
    #     else:
    #         max_s_width = self.style_len

    #     # shape: [N,1,256,256] => pad if needed
    #     style_ref = torch.ones([
    #         len(batch),
    #         batch[0]['style'].shape[0],
    #         batch[0]['style'].shape[1],
    #         max_s_width
    #     ], dtype=torch.float32)
    #     laplace_ref = torch.zeros([
    #         len(batch),
    #         batch[0]['laplace'].shape[0],
    #         batch[0]['laplace'].shape[1],
    #         max_s_width
    #     ], dtype=torch.float32)

    #     wid_list = []
    #     for i, item in enumerate(batch):
    #         cur_w = item['style'].shape[2]  # probably 256
    #         if max_s_width < self.style_len:
    #             style_ref[i, :, :, :cur_w]   = item['style']
    #             laplace_ref[i, :, :, :cur_w] = item['laplace']
    #         else:
    #             style_ref[i, :, :, :cur_w]   = item['style'][:, :, :self.style_len]
    #             laplace_ref[i, :, :, :cur_w] = item['laplace'][:, :, :self.style_len]
    #         wid_list.append(item['wid'])

    #     return {
    #         'style':   style_ref,   # [N,1,256,256]
    #         'laplace': laplace_ref, # [N,1,256,256]
    #         'wid':     wid_list
    #     }


# =======================================
# Prepare the Content Image During Inference
# =======================================
class ContentData(IAMDataset):
    """
    Minimal for text->glyph. Ignores images entirely, so we override __init__ to skip image logic.
    """
    def __init__(self, content_type='unifont_arabic'):
        # letters used for fallback
        self.letters = letters

        # self.letter2index = {label: n for n, label in enumerate(self.letters)}

        # So that hamza is not considered for blank tokens
        self.letter2index = {label: n+1 for n, label in enumerate(self.letters)}

        # load the con_symbols from pickle
        self.con_symbols, self.letter2index = self.get_symbols(content_type)

    def get_content(self, text):
        # shape the text
        indices, _ = shape_arabic_text(text, self.letter2index)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        glyphs = self.con_symbols[indices_tensor]
        glyphs = 1.0 - glyphs
        return glyphs.unsqueeze(0)  # [1, len(text), 16,16]

    # overrides IAMDataset's get_symbols to skip path logic
    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            data = pickle.load(f)
        glyph_entries = data['with_harakat']['glyph_entries']
        letter2index = data['with_harakat']['letter2index']

        max_idx = max(e['idx'][0] for e in glyph_entries)
        glyph_list = [None]*(max_idx+1)

        for entry in glyph_entries:
            idx_val = entry['idx'][0]
            mat_16x16 = entry['mat'].astype(np.float32)
            glyph_list[idx_val] = torch.from_numpy(mat_16x16)
        con_symbols = torch.stack(glyph_list)
        return con_symbols, letter2index
