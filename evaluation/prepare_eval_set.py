import pandas as pd
import os
import shutil
from pathlib import Path

def main():
    # --- Configuration ---
    TEST_TXT = "./data/test.txt"
    SOURCE_FOLDER = "./data/combined_dataset/test"
    OUTPUT_FOLDER = "eval"
    SAMPLE_SIZE = 715
    RANDOM_SEED = 42

    # --- Step 1: Robust line parsing ---
    lines = []
    with open(TEST_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            first_field, rest = line.split(',', 1)
            if ' ' in rest:
                filename = rest.split(' ', 1)[0]
                gt_part = rest.split(' ', 1)[1]
            else:
                filename = rest
                gt_part = ""
            lines.append([first_field, filename, gt_part])

    df = pd.DataFrame(lines, columns=["meta", "filename", "gt"])
    df['dataset'] = df['meta'].str.split('-').str[0]
    df['image_path'] = df['filename'].apply(lambda f: os.path.join(SOURCE_FOLDER, f))

    # --- Step 2: Stratified sample 715 images across datasets ---
    sampled_df = (
        df.groupby("dataset", group_keys=False)
        .apply(lambda x: x.sample(frac=SAMPLE_SIZE / len(df), random_state=RANDOM_SEED))
        .sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    # --- Step 3: Create eval/ folder ---
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # --- Step 4: Copy sampled images to eval/ ---
    for _, row in sampled_df.iterrows():
        src = row['image_path']
        dst = os.path.join(OUTPUT_FOLDER, os.path.basename(src))
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            print(f"File not found: {src}")

    print(f"Sampled {SAMPLE_SIZE} images from 4 datasets into '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()
