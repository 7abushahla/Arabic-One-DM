import warnings
warnings.filterwarnings("ignore", message="256 extra bytes in post.stringData array")

import argparse
import gen_line_images as synthetic
import random
import os
import pickle
import numpy as np
import glob
from fontTools.ttLib import TTFont
import uuid
import multiprocessing
import re
from tqdm import tqdm  # progress bar library
import time
import logging

# Set the global seed to 42 for reproducibility.
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 40

# --- Helper to filter Arabic words (only basic letters, no diacritics, punctuation, or digits) ---
# Only basic Arabic letters (from U+0621 to U+064A)
arabic_letters_pattern = re.compile(r'^[\u0621-\u064A]+$')

def is_arabic_word(word):
    """
    Returns True if the word consists solely of basic Arabic letters (excluding diacritics, punctuation, and digits)
    and is longer than one character.
    """
    return len(word) > 1 and bool(arabic_letters_pattern.match(word))

def sample_fonts_and_words(all_fonts, all_words, dataset_size):
    """
    Randomly sample dataset_size unique fonts and dataset_size unique words.
    """
    if len(all_fonts) < dataset_size:
        raise ValueError(f"Not enough fonts available. Required {dataset_size}, found {len(all_fonts)}")
    if len(all_words) < dataset_size:
        raise ValueError(f"Not enough words available. Required {dataset_size}, found {len(all_words)}")
    selected_fonts = random.sample(all_fonts, dataset_size)
    selected_words = random.sample(all_words, dataset_size)
    return selected_fonts, selected_words

def save_img_text(img, font, output_path, unique_suffix):
    """
    Save the image using a file name that includes the font name and a unique suffix.
    Returns the file name on success.
    """
    base_font = os.path.splitext(os.path.basename(font))[0]
    img_filename = os.path.join(output_path, f"{base_font}_{unique_suffix}.jpg")
    img.convert('RGB').save(img_filename)
    return img_filename

def try_fix_rendered_text(rendered_text, font, font_size, all_backgrounds, image_height):
    """
    Try to fix rendered_text by replacing one character at a time with a space.
    Returns a tuple (fixed_text, img) if successful, or (None, None) if not.
    """
    for char in rendered_text:
        try:
            modified_text = rendered_text.replace(char, ' ')
            img, _ = synthetic.main_generate_image(
                modified_text, font, font_size, all_backgrounds,
                ht=image_height * 5, distort_chance=0.05, blur_chance=0.3, flip=False
            )
            if img is not None:
                return modified_text, img
        except Exception:
            continue
    return None, None

def create_placeholder_image(font, image_height):
    """
    Creates a placeholder image with a white background and black Arabic text.
    A random fallback Arabic word is chosen (e.g., "كلمة", "مثال", "اختبار").
    """
    from PIL import ImageDraw, ImageFont, Image
    import random
    width = image_height * 2
    img = Image.new("RGB", (width, image_height), color="white")
    draw = ImageDraw.Draw(img)
    try:
        basic_font = ImageFont.truetype(font=font, size=image_height // 2)
    except Exception:
        basic_font = ImageFont.load_default()
    fallback_word = random.choice(["كلمة", "مثال", "اختبار"])
    text_width, text_height = draw.textsize(fallback_word, font=basic_font)
    position = ((width - text_width) // 2, (image_height - text_height) // 2)
    draw.text(position, fallback_word, fill="black", font=basic_font)
    return img

def worker_process_groups(group_list, all_backgrounds, font_size, image_height, output_path, seed, pos, 
                            failed_fonts_gen, progress_counter, selected_words, max_attempts=3, max_fallback_attempts=3):
    """
    Processes a list of (word, font, index) tuples.
    Each tuple is processed exactly once by calling process_tuple().
    A dedicated logger (logging warnings/errors only) is used.
    """
    # Setup a dedicated logger for this worker (only logging failures)
    logger = logging.getLogger(f"worker_{pos}")
    logger.setLevel(logging.WARNING)
    fh = logging.FileHandler(f"worker_{pos}.log", mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    random.seed(seed)
    np.random.seed(seed)
    output_dict = {}

    # Local function to process a single tuple.
    def process_tuple(word, font, idx):
        try:
            font_temp = TTFont(font)
            cmap = font_temp['cmap'].getBestCmap()
        except Exception as e:
            logger.error(f"Error reading font {font}: {e}")
            return None, None

        if cmap is None:
            logger.error(f"No cmap for font {os.path.basename(font)}.")
            return None, None

        # Primary attempts
        current_word = word
        for attempt in range(1, max_attempts+1):
            try:
                rendered_text = "".join([e for e in current_word if ord(e) in cmap])
            except Exception as e:
                logger.error(f"Error processing word '{current_word}' for font {font}: {e}")
                continue
            if len(rendered_text) == 0:
                logger.warning(f"Rendered text empty for '{current_word}' with font {font} on primary attempt {attempt}.")
                continue
            try:
                img, _ = synthetic.main_generate_image(
                    rendered_text, font, font_size, all_backgrounds,
                    ht=image_height * 5, distort_chance=0.05, blur_chance=0.3, flip=False
                )
            except Exception as e:
                logger.error(f"Exception during image generation for '{rendered_text}' with font {font}: {e}")
                fixed_text, img = try_fix_rendered_text(rendered_text, font, font_size, all_backgrounds, image_height)
                if fixed_text is not None and img is not None:
                    current_word = fixed_text
                else:
                    continue
            if img is None:
                logger.warning(f"Image generation returned None for '{rendered_text}' with font {font} on primary attempt {attempt}.")
                continue
            return rendered_text, img

        # Fallback attempts
        for fallback in range(1, max_fallback_attempts+1):
            current_word = random.choice(selected_words)
            try:
                rendered_text = "".join([e for e in current_word if ord(e) in cmap])
            except Exception as e:
                logger.error(f"Fallback error processing word '{current_word}' for font {font}: {e}")
                continue
            if len(rendered_text) == 0:
                logger.warning(f"Fallback: Rendered text empty for '{current_word}' with font {font} on fallback attempt {fallback}.")
                continue
            try:
                img, _ = synthetic.main_generate_image(
                    rendered_text, font, font_size, all_backgrounds,
                    ht=image_height * 5, distort_chance=0.05, blur_chance=0.3, flip=False
                )
            except Exception as e:
                logger.error(f"Fallback exception during image generation for '{rendered_text}' with font {font}: {e}")
                fixed_text, img = try_fix_rendered_text(rendered_text, font, font_size, all_backgrounds, image_height)
                if fixed_text is not None and img is not None:
                    rendered_text = fixed_text
                else:
                    continue
            if img is None:
                logger.warning(f"Fallback: Image generation returned None for '{rendered_text}' with font {font} on fallback attempt {fallback}.")
                continue
            return rendered_text, img

        return None, None  # All attempts failed.

    # Process each tuple exactly once.
    for word, font, idx in group_list:
        rendered_text, img = process_tuple(word, font, idx)
        if img is None:
            logger.error(f"Failed to generate image for (word: {word}, font: {font}). Using placeholder.")
            try:
                img = create_placeholder_image(font, image_height)
            except Exception as e_save:
                logger.error(f"Error creating placeholder image for font {font}: {e_save}")
                with progress_counter.get_lock():
                    progress_counter.value += 1
                continue
        # Create a globally unique suffix with worker id, tuple index, and a short UUID.
        unique_suffix = f"{os.path.splitext(os.path.basename(font))[0]}_{pos}_{idx}_{uuid.uuid4().hex[:8]}"
        try:
            file_name = save_img_text(img, font, output_path, unique_suffix)
        except Exception as e_save:
            logger.error(f"Error saving image for (word: {word}, font: {font}): {e_save}")
            with progress_counter.get_lock():
                progress_counter.value += 1
            continue
        output_dict[file_name] = os.path.basename(font)
        with progress_counter.get_lock():
            progress_counter.value += 1

    os.makedirs("word_dict", exist_ok=True)
    dict_filename = os.path.join("word_dict", f"{uuid.uuid4().hex}.pkl")
    with open(dict_filename, 'wb') as f:
        pickle.dump(output_dict, f)
    logger.error(f"Process (seed {seed}, pos {pos}) finished generating {len(output_dict)} images and saved dictionary as {dict_filename}.")

def main(args):
    # Delete old worker log files.
    log_files = glob.glob(os.path.join(os.getcwd(), "worker_*.log"))
    for log_file in log_files:
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"Error deleting log file {log_file}: {e}")

    # Define output paths.
    OUTPUT_PATH = os.path.join("word_images", args.data_dir)
    FONT_PATH = './fonts/'
    BACKGROUND_FOLDER = './images/'

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs("word_dict", exist_ok=True)

    # Clean old files.
    for f in glob.glob(os.path.join(OUTPUT_PATH, "*")):
        os.remove(f)
    for f in glob.glob(os.path.join("word_dict", "*")):
        os.remove(f)

    FONT_SIZE = 80
    IMAGE_HT = 64

    BACKGROUND_FOLDER_NAME = glob.glob(BACKGROUND_FOLDER + "*.png")

    # Load all words.
    with open('./words.pickle', 'rb') as handle:
        all_words = pickle.load(handle)

    # Filter to only Arabic words (and longer than one character) and exclude Arabic digits.
    all_words = [w for w in all_words if is_arabic_word(w)]
    if len(all_words) < args.dataset_size:
        raise ValueError(f"Not enough Arabic words available. Required {args.dataset_size}, found {len(all_words)}.")

    # Get list of fonts.
    all_fonts = synthetic.get_fonts(FONT_PATH, "")
    random.shuffle(all_fonts)
    if len(all_fonts) < args.dataset_size:
        raise ValueError(f"Not enough fonts available. Required {args.dataset_size}, found {len(all_fonts)}.")

    # Exclude fonts that previously failed generation (if file exists).
    failed_fonts_file = "failed_fonts.txt"
    failed_fonts_set = set()
    if os.path.exists(failed_fonts_file):
        try:
            with open(failed_fonts_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        failed_fonts_set.add(line)
            print("Failed fonts file found. Excluding fonts:", failed_fonts_set)
        except Exception as e:
            print("Error reading failed fonts file:", e)
    else:
        print("Failed fonts file not found. Proceeding without excluding any fonts.")

    all_fonts = [font for font in all_fonts if os.path.basename(font) not in failed_fonts_set]

    # Sample exactly dataset_size unique fonts and words.
    dataset_size = args.dataset_size
    selected_fonts, selected_words = sample_fonts_and_words(all_fonts, all_words, dataset_size)
    print(f"Initially sampled {len(selected_fonts)} fonts and {len(selected_words)} words.")
    
    # Export the selected words to a text file.
    with open("selected_words.txt", "w", encoding="utf-8") as f:
        for word in selected_words:
            clean_word = word.strip()
            if clean_word:
                f.write(clean_word + "\n")
    print("Exported selected words to 'selected_words.txt'.")

    # Validate fonts and replace any that fail.
    valid_fonts = []
    failed_fonts = []
    all_arabic_letters = [chr(c) for c in range(0x0621, 0x064B)]
    for font in selected_fonts:
        try:
            font_temp = TTFont(font)
            cmap = font_temp['cmap'].getBestCmap()
        except Exception as e:
            print(f"Error reading font {font}: {e}")
            failed_fonts.append(font)
            continue
        if cmap is None:
            print(f"No cmap for font {font}.")
            failed_fonts.append(font)
            continue
        font_valid = True
        for letter in all_arabic_letters:
            if ord(letter) not in cmap:
                font_valid = False
                print(f"Font {font} is missing letter {letter} in cmap.")
                break
            glyph_name = cmap[ord(letter)]
            if "glyf" in font_temp:
                if glyph_name not in font_temp["glyf"].glyphs:
                    font_valid = False
                    print(f"Font {font} is missing bitmap for glyph {glyph_name} corresponding to letter {letter}.")
                    break
                else:
                    num_contours = getattr(font_temp["glyf"].glyphs[glyph_name], "numberOfContours", None)
                    if not num_contours or num_contours == 0:
                        font_valid = False
                        print(f"Font {font} has empty glyph for letter {letter} (glyph name: {glyph_name}).")
                        break
        if not font_valid:
            failed_fonts.append(font)
            continue

        tests = random.sample(selected_words, 2)
        valid = True
        for word in tests:
            try:
                rendered_text = "".join([e for e in word if ord(e) in cmap])
            except Exception as e:
                valid = False
                break
            if len(rendered_text) == 0:
                valid = False
                break
        if valid:
            valid_fonts.append(font)
        else:
            print(f"Font {font} failed validation with test words.")
            failed_fonts.append(font)
    while len(valid_fonts) < dataset_size:
        remaining_fonts = [f for f in all_fonts if f not in valid_fonts]
        if not remaining_fonts:
            raise ValueError("Not enough valid fonts available.")
        new_font = random.choice(remaining_fonts)
        try:
            font_temp = TTFont(new_font)
            cmap = font_temp['cmap'].getBestCmap()
        except Exception as e:
            continue
        if cmap is None:
            continue
        tests = random.sample(selected_words, 2)
        valid = True
        for word in tests:
            try:
                rendered_text = "".join([e for e in word if ord(e) in cmap])
            except Exception as e:
                valid = False
                break
            if len(rendered_text) == 0:
                valid = False
                break
        if valid:
            valid_fonts.append(new_font)
            print(f"Added replacement font {new_font}")
    selected_fonts = valid_fonts[:dataset_size]
    print(f"Final set of fonts: {[os.path.basename(f) for f in selected_fonts]}")
    print(f"Selected {len(selected_fonts)} unique fonts and {len(selected_words)} unique words for the dataset.")

    with open("selected_fonts.txt", "w", encoding="utf-8") as f:
        for font in selected_fonts:
            f.write(font + "\n")
    print("Exported selected fonts to 'selected_fonts.txt'.")

    # Build groups: each font gets paired with each word.
    groups = []
    for font in selected_fonts:
        group = [(word, font, i) for i, word in enumerate(selected_words, start=1)]
        groups.append(group)
    total_images = sum(len(g) for g in groups)
    print(f"Total images to generate (Cartesian product): {total_images}")

    # Split groups evenly among processes.
    num_processes = args.processes
    groups_sublists = np.array_split(groups, num_processes)
    sublists = [[tpl for group in sublist for tpl in group] for sublist in groups_sublists]

    manager = multiprocessing.Manager()
    failed_fonts_gen = manager.list()
    progress_counter = multiprocessing.Value('i', 0)

    processes = []
    for i, sublist in enumerate(sublists):
        p = multiprocessing.Process(
            target=worker_process_groups,
            args=(list(sublist), BACKGROUND_FOLDER_NAME, FONT_SIZE, IMAGE_HT, OUTPUT_PATH,
                  GLOBAL_SEED + i, i, failed_fonts_gen, progress_counter, selected_words)
        )
        processes.append(p)

    for p in processes:
        p.start()

    with tqdm(total=total_images, desc="Total Progress", position=0) as progress_bar:
        while any(p.is_alive() for p in processes):
            with progress_counter.get_lock():
                current = progress_counter.value
            progress_bar.n = current
            progress_bar.refresh()
            time.sleep(0.5)
        with progress_counter.get_lock():
            progress_bar.n = progress_counter.value
        progress_bar.refresh()

    for p in processes:
        p.join()

    print("Done with Creation of dataset.")

    jpg_files = glob.glob(os.path.join(OUTPUT_PATH, "*.jpg"))
    total_generated = len(jpg_files)
    print(f"Total generated JPEG files: {total_generated}")
    if total_generated < total_images:
        print(f"WARNING: Only {total_generated} images were generated, but {total_images} were expected.")
        # Only then check per-class counts.
        expected_count = len(selected_words)
        for font in selected_fonts:
            base_font = os.path.splitext(os.path.basename(font))[0]
            font_files = glob.glob(os.path.join(OUTPUT_PATH, f"{base_font}_*.jpg"))
            count_font = len(font_files)
            if count_font < expected_count:
                print(f"Font {base_font}: {count_font} images generated (expected {expected_count}).")
    else:
        print("All images were successfully generated.")

    if failed_fonts_gen:
        print("The following fonts failed during generation:")
        for font in failed_fonts_gen:
            print(font)
        with open("failed_fonts_generation.txt", "w", encoding="utf-8") as f:
            for font in failed_fonts_gen:
                f.write(font + "\n")
        print("Exported failed fonts (during generation) to 'failed_fonts_generation.txt'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_size", type=int, help="Number of unique fonts and words to sample (dataset will be size x size)")
    parser.add_argument("processes", type=int, help="Number of processes to use")
    parser.add_argument("data_dir", help="Name of the output directory under word_images")
    main(parser.parse_args())
