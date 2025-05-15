import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader_ara import ContentData, generate_type
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

class StyleRefDataset(torch.utils.data.Dataset):
    def __init__(self, style_path, laplace_path, writer_to_image, writer_ids):
        self.style_path = style_path
        self.laplace_path = laplace_path
        self.writer_to_image = writer_to_image
        self.writer_ids = writer_ids

    def __len__(self):
        return len(self.writer_ids)

    def __getitem__(self, idx):
        wr_id = self.writer_ids[idx]
        image_name = self.writer_to_image[wr_id]
        s_path = os.path.join(self.style_path, image_name)
        l_path = os.path.join(self.laplace_path, image_name)
        s_img = cv2.imread(s_path, flags=0)
        l_img = cv2.imread(l_path, flags=0)
        if s_img is None or l_img is None:
            raise RuntimeError(f"Error reading style or laplace image for file '{image_name}' (writer '{wr_id}') in {self.style_path}")
        s_img = cv2.resize(s_img, (256, 256), interpolation=cv2.INTER_AREA)
        l_img = cv2.resize(l_img, (256, 256), interpolation=cv2.INTER_AREA)
        style_t = torch.from_numpy(s_img.astype(np.float32) / 255.0).unsqueeze(0)  # [1,256,256]
        laplace_t = torch.from_numpy(l_img.astype(np.float32) / 255.0).unsqueeze(0)  # [1,256,256]
        # Repeat to get [2,256,256] for model compatibility
        style_t = style_t.repeat(2, 1, 1)
        laplace_t = laplace_t.repeat(2, 1, 1)
        return {
            'style': style_t,
            'laplace': laplace_t,
            'wid': wr_id
        }

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

    # Split writers for distributed processing
    each_process = len(writer_ids) // totol_process
    remainder = len(writer_ids) % totol_process
    start_idx = local_rank * each_process + min(local_rank, remainder)
    end_idx = start_idx + each_process + (1 if local_rank < remainder else 0)
    temp_writer_ids = writer_ids[start_idx:end_idx]

    style_dataset = StyleRefDataset(
        os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
        os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]),
        writer_to_image,
        temp_writer_ids
    )
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True)

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

    loader_iter = iter(style_loader)
    for i, writer_id in enumerate(temp_writer_ids):
        data = next(loader_iter)
        style_input = data['style'].to(opt.device)      # [1,2,256,256]
        laplace = data['laplace'].to(opt.device)        # [1,2,256,256]
        wid = data['wid']                               # [1] or string
        for word in words:
            text_ref = load_content.get_content(word)
            text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[3]//8, (text_ref.shape[1]*32)//8)).to(opt.device)

            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, laplace, text_ref,
                                                        opt.sampling_timesteps, opt.eta)
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, laplace, text_ref)
            else:
                raise ValueError('sample method is not supported')

            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
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