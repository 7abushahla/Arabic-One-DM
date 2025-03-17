import argparse
import glob
import tqdm
import pickle
import os

def csv_field(field):
    """Return the CSV field properly quoted if it contains a comma or a double quote."""
    if ',' in field or '"' in field:
        field = field.replace('"', '""')
        return f'"{field}"'
    return field

def main(args):
    outfile = 'data.csv'
    # Set base to args.base_dir; in your case, pass the root (e.g., "/Users/hamza/Research/One-DM")
    base = args.base_dir

    # Define the unwanted prefix that should be removed from each image path.
    unwanted_prefix = os.path.join("test", "dataset") + os.sep

    # Collect all rows in a list.
    rows = []
    total_files = 0
    for pickle_file in glob.glob("word_dict/*pkl"):
        with open(pickle_file, 'rb') as handle:
            data_dict = pickle.load(handle)
            for name, font_label in data_dict.items():
                # Remove the unwanted prefix from the image path, if present.
                if name.startswith(unwanted_prefix):
                    name = name[len(unwanted_prefix):]
                total_files += 1
                # Construct the full path using the base directory.
                path = os.path.join(base, name)
                # Clean the font name: remove file extension and trailing spaces/underscores.
                font_clean = os.path.splitext(os.path.basename(font_label))[0].rstrip(" _")
                rows.append((path, font_clean))
    
    # Sort rows by the font label.
    rows.sort(key=lambda x: x[1])
    
    # Write sorted rows to CSV.
    with open(outfile, 'w', encoding='utf-8') as csvfile:
        csvfile.write("img_path,text\n")
        for path, font_clean in rows:
            csvfile.write(f"{csv_field(path)},{csv_field(font_clean)}\n")
    
    print("Total Files:", total_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", nargs='?', default="")
    main(parser.parse_args())
