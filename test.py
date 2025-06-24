import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader_ara import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
import torch.distributed as dist
from utils.util import fix_seed
import cv2
import numpy as np

def get_writer_to_first_image(test_txt_path):
    writer_to_image = {}
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            first_field = parts[0]
            writer_id, image_name = first_field.split(',')
            if writer_id not in writer_to_image:
                writer_to_image[writer_id] = image_name
    return writer_to_image

def get_words_from_oov(oov_path):
    with open(oov_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

# ---------------------------------------
# Dataset: one reference image per writer
# ---------------------------------------

class StyleRefDataset(torch.utils.data.Dataset):
    """Load *one* style + laplace reference per writer using the mapping
    built from test.txt (get_writer_to_first_image).  Returns a 2-channel
    reference (anchor & positive are identical) so shapes stay compatible
    with the original network.
    """

    def __init__(self, style_root, laplace_root, writer_to_image):
        self.style_root   = style_root
        self.laplace_root = laplace_root
        self.mapping      = writer_to_image               # dict[writer] → imgName
        self.writers      = list(self.mapping.keys())     # stable order

    def __len__(self):
        return len(self.writers)

    def __getitem__(self, idx):
        wid   = self.writers[idx]
        fname = self.mapping[wid]

        s_path = os.path.join(self.style_root, fname)
        l_path = os.path.join(self.laplace_root, fname)

        s_img = cv2.imread(s_path, flags=0)
        l_img = cv2.imread(l_path, flags=0)
        if s_img is None or l_img is None:
            raise RuntimeError(f"Cannot read style/laplace for {fname} (writer {wid})")

        # normalise & convert → tensor, [1,H,W]
        s_t = torch.from_numpy(s_img.astype(np.float32) / 255.0).unsqueeze(0)
        l_t = torch.from_numpy(l_img.astype(np.float32) / 255.0).unsqueeze(0)

        # repeat so network sees 2 refs (anchor+pos)
        s_t = s_t.repeat(2, 1, 1)
        l_t = l_t.repeat(2, 1, 1)

        return {"style": s_t, "laplace": l_t, "wid": wid}

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    load_content = ContentData()
    totol_process = dist.get_world_size()

    # Paths
    test_txt_path = 'data/test.txt'
    oov_words_path = 'data/oov.common_words'

    # Get mapping: writer_id -> first style image
    writer_to_image = get_writer_to_first_image(test_txt_path)
    writer_ids = list(writer_to_image.keys())
    words = get_words_from_oov(oov_words_path)

    print(f"[DEBUG] Total writers found in test.txt: {len(writer_ids)}")
    print(f"[DEBUG] Total words found in oov.common_words: {len(words)}")
    print(f"[DEBUG] Sample writer IDs: {writer_ids[:5]}")
    print(f"[DEBUG] Sample words: {words[:5]}")

    if len(writer_ids) == 0 or len(words) == 0:
        print("[ERROR] No writers or words found. Exiting.")
        return

    # Split words across processes (mirrors original generation logic)
    each_process = len(words) // totol_process
    if len(words) % totol_process == 0:
        temp_words = words[local_rank * each_process:(local_rank + 1) * each_process]
    else:
        each_process += 1
        temp_words = words[local_rank * each_process:(local_rank + 1) * each_process]

    # One reference image per writer (no 6 k aggregation)
    style_dataset = StyleRefDataset(
        os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
        os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]),
        writer_to_image
    )
    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )

    target_dir = os.path.join(opt.save_dir, opt.generate_type)

    diffusion = Diffusion(device=opt.device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(opt.device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(opt.device)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Iterate writer-by-writer so each writer folder is filled with all words at once
    for data in tqdm(style_loader, position=0, desc='writer'):
        style_input   = data['style'].to(opt.device)   # [1,2,H,W]
        laplace_input = data['laplace'].to(opt.device) # [1,2,H,W]
        writer_id     = data['wid'][0] if isinstance(data['wid'], (list, tuple)) else data['wid']

        for word in temp_words:
            text_ref = load_content.get_content(word)
            text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)

            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(opt.device)

            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0], x, style_input, laplace_input, text_ref,
                    opt.sampling_timesteps, opt.eta
                )
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(
                    unet, vae, style_input.shape[0], x, style_input, laplace_input, text_ref
                )
            else:
                raise ValueError('sample method is not supported')

            for img_idx in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[img_idx])
                image = im.convert("L")
                out_path = os.path.join(target_dir, writer_id)
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, word + ".png"))

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', help='target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='', required=True, help='pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True, help='four generation settings:iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cuda', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)