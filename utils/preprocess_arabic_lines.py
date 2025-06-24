import argparse
from pathlib import Path
from multiprocessing.pool import ThreadPool
import cv2

TARGET_H = 64            # fixed height
MIN_W    = 128           # clamp to avoid very narrow lines
MAX_W    = 416           # clamp to avoid extremely wide lines
ROUND_TO = 16            # make width a multiple of 8 or 16
N_THREADS = 8


def resize_keep_h(img):
    """Resize image so that height=TARGET_H and width rounded up to multiple of ROUND_TO."""
    h, w = img.shape[:2]
    new_w = int(round(w * TARGET_H / h))
    new_w = max(MIN_W, min(MAX_W, new_w))
    # round up to next multiple of ROUND_TO
    new_w = (new_w + ROUND_TO - 1) // ROUND_TO * ROUND_TO
    return cv2.resize(img, (new_w, TARGET_H), cv2.INTER_AREA)


def process_one(arg):
    src_path, dst_path = arg
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  could not read {src_path}")
        return
    out = resize_keep_h(img)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), out)


def collect_jobs(root_src: Path, root_dst: Path):
    jobs = []
    for split in ("train", "val", "test"):
        split_src = root_src / split
        if not split_src.exists():
            continue
        for file in split_src.rglob("*.*"):
            rel = file.relative_to(root_src)
            jobs.append((file, root_dst / rel))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Resize Arabic line images to 64px height and pad width to multiple of 16.")
    parser.add_argument("src", nargs="+", help="Source root directories (each must contain train/ val/ test/)" )
    parser.add_argument("--suffix", default="_64", help="Suffix added to create output directory next to each src root")
    parser.add_argument("--threads", type=int, default=N_THREADS, help="Number of parallel threads")
    args = parser.parse_args()

    all_jobs = []
    for src_root in args.src:
        src_path = Path(src_root)
        if not src_path.exists():
            print(f"❌ Directory '{src_path}' not found – skip.")
            continue
        dst_path = Path(src_root + args.suffix)
        all_jobs.extend(collect_jobs(src_path, dst_path))

    print(f"🖼️  {len(all_jobs)} images to process …")
    with ThreadPool(args.threads) as pool:
        for _ in pool.imap_unordered(process_one, all_jobs):
            pass
    print("✅  Done – resized images saved alongside original folders.")


if __name__ == "__main__":
    main() 