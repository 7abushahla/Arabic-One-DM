import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from data_loader.loader_ara import letters
from models.recognition import HTRNet
import numpy as np
from tqdm import tqdm
from evaluation.gs import rlts, geom_score, fancy_plot
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

TARGET_SHAPE = (64, 256)  # Fixed shape for geometry score

def compute_cer(ref: str, hyp: str) -> float:
    import editdistance
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)

def compute_wer(ref: str, hyp: str) -> float:
    import editdistance
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)

def build_idx_to_char():
    return {i+1: ch for i, ch in enumerate(letters)}

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)  # [1,3,256,256]

def decode_ctc(log_probs, idx_to_char):
    preds = torch.argmax(log_probs, dim=2)  # [T, B]
    raw_seq = preds[:, 0].tolist()
    dedup = []
    prev = None
    for ch_idx in raw_seq:
        if ch_idx != prev and ch_idx != 0:
            dedup.append(ch_idx)
        prev = ch_idx
    recognized = "".join(idx_to_char.get(ch, "?") for ch in dedup)
    return recognized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True, help='Path to generated images root (e.g. Generated/oov_u)')
    parser.add_argument('--ocr_model', type=str, default='./models/ocr_best_state_recognition_OG.pth', help='Path to OCR model weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    idx_to_char = build_idx_to_char()

    # Build and load OCR model
    ocr_model = HTRNet(nclasses=len(letters)+1, vae=False)
    state_dict = torch.load(args.ocr_model, map_location='cpu')
    ocr_model.load_state_dict(state_dict)
    ocr_model = ocr_model.to(device)
    ocr_model.eval()
    print(f"[INFO] Loaded OCR model from {args.ocr_model}")

    writer_ids = [d for d in os.listdir(args.gen_dir) if os.path.isdir(os.path.join(args.gen_dir, d))]
    print(f"[DEBUG] Total writers found: {len(writer_ids)}")

    # Collect all words
    all_words = set()
    for writer_id in writer_ids:
        writer_path = os.path.join(args.gen_dir, writer_id)
        for fname in os.listdir(writer_path):
            if fname.endswith('.png'):
                word = os.path.splitext(fname)[0]
                all_words.add(word)
    print(f"[DEBUG] Total unique words found: {len(all_words)}")

    all_cer = []
    all_wer = []
    total = 0
    for writer_id in tqdm(writer_ids, desc='Writers'):
        writer_path = os.path.join(args.gen_dir, writer_id)
        for fname in os.listdir(writer_path):
            if not fname.endswith('.png'):
                continue
            word = os.path.splitext(fname)[0]
            img_path = os.path.join(writer_path, fname)
            img_tensor = preprocess_image(img_path).to(device)
            with torch.no_grad():
                logits = ocr_model(img_tensor)  # [T, B, C]
                log_probs = F.log_softmax(logits, dim=2)
            recognized = decode_ctc(log_probs, idx_to_char)
            cer = compute_cer(word, recognized)
            wer = compute_wer(word, recognized)
            all_cer.append(cer)
            all_wer.append(wer)
            total += 1
    avg_cer = np.mean(all_cer) if all_cer else 0.0
    std_cer = np.std(all_cer) if all_cer else 0.0
    avg_wer = np.mean(all_wer) if all_wer else 0.0
    std_wer = np.std(all_wer) if all_wer else 0.0
    print(f"Total samples: {total}")
    print(f"Average CER: {avg_cer:.4f} ± {std_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f} ± {std_wer:.4f}")

    # --- GeomScore Evaluation ---
    print("\n[INFO] Computing GeomScore between generated and real eval set...")

    def load_and_flatten_images(folder, limit=None):
        arrs = []
        count = 0
        image_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(image_exts):
                continue
            img = imread(os.path.join(folder, fname), as_gray=True)
            anti_alias = False if img.dtype == bool else True
            img_resized = resize(img, TARGET_SHAPE, anti_aliasing=anti_alias)
            arr = img_resized.astype('float32') / 255.0
            arrs.append(arr.flatten())
            count += 1
            if limit and count >= limit:
                break
        D = TARGET_SHAPE[0] * TARGET_SHAPE[1]
        return np.stack(arrs) if arrs else np.zeros((0, D))

    # Load generated images
    gen_imgs = []
    image_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
    for writer_id in writer_ids:
        writer_path = os.path.join(args.gen_dir, writer_id)
        for fname in os.listdir(writer_path):
            if not fname.lower().endswith(image_exts):
                continue
            img = imread(os.path.join(writer_path, fname), as_gray=True)
            anti_alias = False if img.dtype == bool else True
            img_resized = resize(img, TARGET_SHAPE, anti_aliasing=anti_alias)
            arr = img_resized.astype('float32') / 255.0
            gen_imgs.append(arr.flatten())
    D = TARGET_SHAPE[0] * TARGET_SHAPE[1]
    X_gen = np.stack(gen_imgs) if gen_imgs else np.zeros((0, D))
    print(f"[DEBUG] Generated dataset path: {args.gen_dir}")
    print(f"[DEBUG] Number of generated images: {X_gen.shape[0]}")

    # Load real images from evaluation/eval/
    eval_dir = os.path.join('evaluation', 'eval')
    X_real = load_and_flatten_images(eval_dir, limit=X_gen.shape[0])
    print(f"[DEBUG] Real dataset path: {eval_dir}")
    print(f"[DEBUG] Number of real images: {X_real.shape[0]}")

    # Shape check
    if X_real.shape != X_gen.shape:
        raise ValueError(f"Shape mismatch: X_real {X_real.shape} vs X_gen {X_gen.shape}. "
                         f"Ensure both are the same number of samples and dimensions.")

    # Compute RLTs
    rlt_gen = rlts(X_gen, n=100, L_0=32, i_max=100, gamma=1.0/8)
    rlt_real = rlts(X_real, n=100, L_0=32, i_max=100, gamma=1.0/8)

    # Plot MRLTs
    # Generated MRLT
    plt.figure()
    mean_gen = np.mean(rlt_gen, axis=0)
    fancy_plot(mean_gen, label='Generated MRLT', color='C0')
    plt.xlim([0, 30])
    plt.legend()
    plt.title('MRLT of Generated')
    plt.savefig('MRLT_generated.png')
    plt.close()

    # Real MRLT
    plt.figure()
    mean_real = np.mean(rlt_real, axis=0)
    fancy_plot(mean_real, label='Real MRLT', color='C1')
    plt.xlim([0, 30])
    plt.legend()
    plt.title('MRLT of Real')
    plt.savefig('MRLT_real.png')
    plt.close()

    score = geom_score(rlt_gen, rlt_real)
    print(f"GeomScore: {score:.4f}")

if __name__ == '__main__':
    main() 