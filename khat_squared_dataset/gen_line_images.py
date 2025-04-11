import warnings
warnings.filterwarnings("ignore", message="256 extra bytes in post.stringData array")

import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import time
import cv2
import image_distortion
import re

# --- New helper to filter Arabic words ---
# Only Arabic letters (from U+0621 to U+064A), excluding diacritics, punctuation, and digits.
arabic_letters_pattern = re.compile(r'^[\u0621-\u064A]+$')

def is_arabic_word(word):
    """
    Returns True if the word consists solely of basic Arabic letters (excluding diacritics, punctuation, and digits)
    and is longer than one character.
    """
    return len(word) > 1 and bool(arabic_letters_pattern.match(word))

# Global parameters
SEED = 42
MIN_WORDS_IN_LINE = 12
MAX_WORDS_IN_LINE = 25
MAX_SPECIAL_IN_LINE = 0

FONT_PATH = './fonts_otf/'
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 40
BUFFER_x = 5  # Buffer to leave in line images (x-axis)
BUFFER_y = 5  # Buffer (y-axis)
IMAGE_HT = 64  # Image height to save

# def get_fonts(font_path=None, type='regular'):
#     if font_path is None:
#         font_path = FONT_PATH
#     font_files = []
#     for dname, dirs, files in os.walk(font_path):
#         for fname in files:
#             # Check for .ttf or .otf extension (case-insensitive) and that the 'type' is in the filename,
#             # and that "Condensed" is not in the filename.
#             if fname.lower().endswith(('ttf', 'otf')) and type in fname.lower() and 'condensed' not in fname.lower():
#                 font_files.append(os.path.join(dname, fname))
#     return font_files

def get_fonts(font_path=None, type='regular'):
    """
    Walks through the font_path (or FONT_PATH if None) and returns a list of font file paths.
    It accepts both .ttf and .otf files (case-insensitive) that contain the string `type`
    in their filename and do not include 'condensed'. If the same font (by base name) exists
    in both .otf and .ttf formats in the same folder, only the .ttf version is kept.
    """
    if font_path is None:
        font_path = './fonts_otf/'
    font_dict = {}
    for dname, dirs, files in os.walk(font_path):
        for fname in files:
            fname_lower = fname.lower()
            # Only consider .ttf or .otf files.
            if not fname_lower.endswith(('ttf', 'otf')):
                continue
            # Filter by desired type and exclude "condensed"
            if type not in fname_lower or 'condensed' in fname_lower:
                continue
            full_path = os.path.join(dname, fname)
            # Use the base name (without extension) as key, in lower case.
            base = os.path.splitext(fname)[0].lower()
            ext = os.path.splitext(fname)[1].lower()  # either ".ttf" or ".otf"
            # If a font with the same base name is already stored, choose the .ttf version if available.
            if base in font_dict:
                existing_ext = os.path.splitext(os.path.basename(font_dict[base]))[1].lower()
                if existing_ext == '.otf' and ext == '.ttf':
                    font_dict[base] = full_path
            else:
                font_dict[base] = full_path
    return list(font_dict.values())


def set_buffers(x, y):
    global BUFFER_x, BUFFER_y
    BUFFER_x = x
    BUFFER_y = y

def get_font(font_list, min_size=25, max_size=40):
    size = np.random.randint(min_size, max_size)
    ind = np.random.randint(0, len(font_list) - 1)
    return font_list[ind], size
    
def create_full_line_img(text, font, font_size, img_ht=IMAGE_HT, flip=False):
    # Load the font and compute the text bounding box.
    image_font = ImageFont.truetype(font=font, size=font_size)
    (left, top, right, bottom) = image_font.getbbox(text)
    abs_left = np.abs(left)
    abs_top = np.abs(top)
    offset_x = abs_left if left < 0 else 0
    offset_y = abs_top if top < 0 else 0
    width = np.abs(right - left) + abs_left
    ht = np.abs(bottom - top) + abs_top
    # Create a white canvas and draw the text.
    txt_img = Image.new('L', (width + 2 * BUFFER_x, ht + 2 * BUFFER_y), color='white')
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((offset_x + BUFFER_x, offset_y + BUFFER_y), text, fill=0, font=image_font)
    txt_img = txt_img.crop((abs_left - offset_x, abs_top - offset_y, 
                            right + offset_x + 2 * BUFFER_x, 
                            bottom + offset_y + 2 * BUFFER_y))
    # --- NEW: add extra padding so characters are not touching the edges ---
    EXTRA_PADDING = 10
    txt_img = ImageOps.expand(txt_img, border=EXTRA_PADDING, fill=255)
    # ---------------------------------------------------------------
    w, h = txt_img.size
    if h == 0:
        return None, 0
    new_width = int(w / h * img_ht)
    txt_img = txt_img.resize((new_width, img_ht))
    if flip:
        txt_img = ImageOps.mirror(txt_img)
    return txt_img, new_width

def generate_one_line(text, fonts_list, add_special=True, img_ht=IMAGE_HT, flip=False):
    font = fonts_list[np.random.randint(0, len(fonts_list))]
    font_size = np.random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE + 1)
    # If add_special is enabled, you might modify text further.
    # (Assuming add_special_to_text is defined elsewhere if needed.)
    # if add_special:
    #     text = add_special_to_text(text, font)
    img, new_width = create_full_line_img(text, font, font_size, img_ht, flip=flip)
    return img, (new_width, img_ht), text, font, font_size

def save_img_text(img, corpus_dir, dir_number, img_number, text):
    # DIR_NAMES, TOTAL_DIGITS_DIR, TOTAL_DIGITS_FILES, and OUTPUT_PATH 
    # should be defined globally or passed as parameters.
    # Here we assume they exist; adjust as needed.
    dir_name = DIR_NAMES[corpus_dir] + '_' + str(dir_number).zfill(TOTAL_DIGITS_DIR)
    dir_path = os.path.join(OUTPUT_PATH, dir_name)
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Created dir', dir_path)
    img_filename = os.path.join(dir_path, dir_name + '_' + str(img_number).zfill(TOTAL_DIGITS_FILES) + '.jpg')
    img.save(img_filename)
    txt_filename = img_filename[:-3] + 'txt'
    with open(txt_filename, 'w') as fout:
        fout.write(text)

def get_line_img_from_file(img_file, poly, new_ht=IMAGE_HT):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = np.array(poly)
    [min_x, min_y] = np.min(pts, axis=0)
    [max_x, max_y] = np.max(pts, axis=0)
    cropped_img = img[min_y:max_y+1, min_x:max_x+1]
    ht, width = cropped_img.shape
    new_width = int(width / ht * new_ht)
    resized_img = cv2.resize(cropped_img, (new_width, new_ht))
    return resized_img, (new_width, new_ht)
                
def read_line_img_from_file(img_file, new_ht=IMAGE_HT):
    if not os.path.isfile(img_file):
        return None, None
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ht, width = img.shape
    new_width = int(width / ht * new_ht)
    resized_img = cv2.resize(img, (new_width, new_ht))
    return resized_img, (new_width, new_ht)

def draw_text_on_background(text, font, font_size, offset=None, background_file=None, 
                              background_img=None, color=None, resized_ht=None, resized_width=None):
    if background_img is None:
        background_img = Image.open(background_file)
    image_font = ImageFont.truetype(font=font, size=font_size)
    background_size = background_img.size

    if np.random.rand() < 0.8:  # 80% chance black
        color = (0, 0, 0)
    else:  # 20% chance gray
        gray_value = np.random.randint(50, 150)  # medium-dark gray
        color = (gray_value, gray_value, gray_value)
    
    (left, top, right, bottom) = image_font.getbbox(text)
    width = np.abs(right - left)
    ht = np.abs(bottom - top)
    if width == 0 or ht == 0:
        return None
    font_x = -left
    font_y = -top
    if background_size[0] < width or background_size[1] < ht:
        background_img = background_img.resize((width, ht))
        background_size = background_img.size
    if offset is None:
        valid_left = np.random.randint(background_size[0] - width + 1)
        valid_top = np.random.randint(background_size[1] - ht + 1)
    offset = (font_x + valid_left, font_y + valid_top)
    draw = ImageDraw.Draw(background_img)
    draw.text(offset, text, font=image_font, fill=color)
    background_img = background_img.crop((valid_left, valid_top, valid_left + width, valid_top + ht))
    width, ht = background_img.size
    if width == 0 or ht == 0:
        return None
    if resized_ht is not None and resized_width is None:
        resized_width = int(width / ht * resized_ht)
    if resized_ht is not None:
        background_img = background_img.resize((resized_width, resized_ht))
    return background_img

# def apply_blur(img, radius=None):
#     if radius is None:
#         radius = np.random.randint(10) / 10 + 1
#     img = img.filter(ImageFilter.GaussianBlur(radius=radius))
#     return img

def apply_blur(img, radius=None):
    if radius is None:
        radius = np.random.uniform(0.5, 2.0)  # subtle blur from 0.5 to 2.0
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def main_generate_image(text, font, font_size, background_file, 
                        distort_chance=0.3, blur_chance=0.5, ht=60, flip=False):
    img_blurred = False
    img_distorted = False
    distortion = 'None'
    if isinstance(background_file, list):
        background = np.random.choice(background_file)
    else:
        background = background_file
    img = draw_text_on_background(text, font, font_size, background_file=background, resized_ht=ht)
    if img is None:
        return img, None
    blur = np.random.rand()
    if blur < blur_chance:
        img = apply_blur(img)
        img_blurred = True
    distort = np.random.rand()
    if distort < distort_chance:
        img, distortion = image_distortion.apply_random_transform(np.array(img))
        img = Image.fromarray(img)
        img_distorted = True
    if flip:
        img = ImageOps.mirror(img)
    return img, {'blurred': img_blurred, 'distorted': img_distorted, 'distortion': distortion}
