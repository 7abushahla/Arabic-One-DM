# import argparse
# import os
# from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
# import torch
# from data_loader.loader_ara import Random_StyleIAMDataset, ContentData, generate_type
# from models.unet import UNetModel
# from tqdm import tqdm
# from diffusers import AutoencoderKL
# from models.diffusion import Diffusion
# import torchvision
# import torch.distributed as dist
# from utils.util import fix_seed

# def main(opt):
#     """ load config file into cfg"""
#     cfg_from_file(opt.cfg_file)
#     assert_and_infer_cfg()
#     """fix the random seed"""
#     fix_seed(cfg.TRAIN.SEED)

#     """ set mulit-gpu """
#     dist.init_process_group(backend='nccl')
#     local_rank = dist.get_rank()
#     torch.cuda.set_device(local_rank)

#     load_content = ContentData()
#     totol_process = dist.get_world_size()

#     text_corpus = generate_type[opt.generate_type][1]
#     with open(text_corpus, 'r') as _f:
#         texts = _f.read().split()
#     each_process = len(texts)//totol_process

#     """split the data for each GPU process"""
#     if  len(texts)%totol_process == 0:
#         temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]
#     else:
#         each_process += 1
#         temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]

    
#     """setup data_loader instances"""
#     style_dataset = Random_StyleIAMDataset(os.path.join(cfg.DATA_LOADER.STYLE_PATH,generate_type[opt.generate_type][0]), 
#                                            os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]), len(temp_texts))
    
#     # style_dataset = Random_StyleIAMDataset(
#     # os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
#     # os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]),
#     # len(temp_texts),
#     # "data/oov.common_words"  # <-- Pass the text_file argument here
#     # )
#     print('this process handle characters: ', len(style_dataset))
#     style_loader = torch.utils.data.DataLoader(style_dataset,
#                                                 batch_size=1,
#                                                 shuffle=True,
#                                                 drop_last=False,
#                                                 num_workers=cfg.DATA_LOADER.NUM_THREADS,
#                                                 pin_memory=True
#                                                 )

#     target_dir = os.path.join(opt.save_dir, opt.generate_type)

#     diffusion = Diffusion(device=opt.device)

#     """build model architecture"""
#     unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
#                      out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
#                      attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
#                      context_dim=cfg.MODEL.EMB_DIM).to(opt.device)
    
#     """load pretrained one_dm model"""
#     if len(opt.one_dm) > 0: 
#         unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=torch.device('cpu')))
#         print('load pretrained one_dm model from {}'.format(opt.one_dm))
#     else:
#         raise IOError('input the correct checkpoint path')
#     unet.eval()

#     vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
#     vae = vae.to(opt.device)
#     # Freeze vae and text_encoder
#     vae.requires_grad_(False)


#     """generate the handwriting datasets"""
#     loader_iter = iter(style_loader)
#     for x_text in tqdm(temp_texts, position=0, desc='batch_number'):
#         data = next(loader_iter)
#         data_val, laplace, wid = data['style'][0], data['laplace'][0], data['wid']
        
#         data_loader = []
#         # split the data into two parts when the length of data is two large
#         if len(data_val) > 224:
#             data_loader.append((data_val[:224], laplace[:224], wid[:224]))
#             data_loader.append((data_val[224:], laplace[224:], wid[224:]))
#         else:
#             data_loader.append((data_val, laplace, wid))
#         for (data_val, laplace, wid) in data_loader:
#             style_input = data_val.to(opt.device)
#             laplace = laplace.to(opt.device)
#             text_ref = load_content.get_content(x_text)
#             text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)
#             x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(opt.device)
            
#             if opt.sample_method == 'ddim':
#                 ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
#                                                         x, style_input, laplace, text_ref,
#                                                         opt.sampling_timesteps, opt.eta)
#             elif opt.sample_method == 'ddpm':
#                 ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
#                                                         x, style_input, laplace, text_ref)
#             else:
#                 raise ValueError('sample method is not supported')
            
#             for index in range(len(ema_sampled_images)):
#                 im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
#                 image = im.convert("L")
#                 out_path = os.path.join(target_dir, wid[index][0])
#                 os.makedirs(out_path, exist_ok=True)
#                 image.save(os.path.join(out_path, x_text + ".png"))

# if __name__ == '__main__':
#     """Parse input arguments"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
#                         help='Config file for training (and optionally testing)')
#     parser.add_argument('--dir', dest='save_dir', default='Generated', help='target dir for storing the generated characters')
#     parser.add_argument('--one_dm', dest='one_dm', default='', required=True, help='pre-train model for generating')
#     parser.add_argument('--generate_type', dest='generate_type', required=True, help='four generation settings:iv_s, iv_u, oov_s, oov_u')
#     parser.add_argument('--device', type=str, default='cuda', help='device for test')
#     parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
#     parser.add_argument('--sampling_timesteps', type=int, default=50)
#     parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
#     parser.add_argument('--eta', type=float, default=0.0)
#     parser.add_argument('--local_rank', type=int, default=0, help='device for training')
#     opt = parser.parse_args()
#     main(opt)


#!/usr/bin/env python
"""
Fully inference‑only version of test.py that eliminates CUDA OOM.
Key safeguards
──────────────
1.   Global `torch.set_grad_enabled(False)` **plus** outer `torch.inference_mode()`
     ensure autograd never stores activations.
2.   Gradient‑checkpointing disabled on every UNet block.
3.   Optional `--fp16` mixed precision.
4.   Shorter chunk length (128) to cap width.

Usage example
─────────────
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 test.py \
    --one_dm ./Saved/IAM64_scratch/Arabic-20250413_145740/model/499-ckpt.pt \
    --generate_type oov_u --dir ./Generated/Arabic --fp16
"""

import argparse, os, contextlib, torch, torch.distributed as dist, torchvision
from tqdm import tqdm
from diffusers import AutoencoderKL

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader_ara import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from models.diffusion import Diffusion
from utils.util import fix_seed

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Silence *all* tqdm calls made **after** this point (i.e. inside diffusion)
# but keep a handle to the original tqdm so we can show our own outer bar.
from functools import partial
import tqdm as _tqdm  # noqa: E402  (must precede diffusion import)
_outer_tqdm = _tqdm.tqdm  # save original

def _silent_tqdm(*args, **kwargs):
    kwargs.setdefault("disable", True)
    return _outer_tqdm(*args, **kwargs)

if not getattr(_tqdm, "_patched_silent", False):
    _tqdm.tqdm = _silent_tqdm  # every later `from tqdm import tqdm` gets silent version
    _tqdm._patched_silent = True
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Disable autograd everywhere before any layer is built
torch.set_grad_enabled(False)

# -----------------------------------------------------------------------------

def main(opt):
    # ---------------- config & env ------------------------------------------
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device)

    # ---------------- text split across ranks -------------------------------
    corpus_file = generate_type[opt.generate_type][1]
    with open(corpus_file) as f:
        words = f.read().split()
    world = dist.get_world_size()
    per_rank = (len(words) + world - 1) // world
    my_words = words[local_rank * per_rank : (local_rank + 1) * per_rank]

    # ---------------- dataloader --------------------------------------------
    style_dataset = Random_StyleIAMDataset(
        os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
        os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]),
        len(my_words),
    )
    print(f"Rank {local_rank}: handling {len(style_dataset)} characters")

    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=True,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True,
    )

    # ---------------- models -------------------------------------------------
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM,
    ).to(device)

    if not opt.one_dm:
        raise IOError("--one_dm checkpoint missing")
    unet.load_state_dict(torch.load(opt.one_dm, map_location="cpu"))
    unet.eval()

    # **turn off custom gradient‑checkpointing flags inside every block**
    for m in unet.modules():
        if hasattr(m, "checkpoint"):
            m.checkpoint = False

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae").to(device)
    vae.eval()

    diffusion = Diffusion(device=device)

    # ---------- output dir ---------------------------------------------------
    out_root = os.path.join(opt.save_dir, opt.generate_type)
    os.makedirs(out_root, exist_ok=True)

    # ---------- helpers ------------------------------------------------------
    autocast_ctx = (
        contextlib.nullcontext()
        if not opt.fp16
        else torch.autocast("cuda", dtype=torch.float16)
    )

    loader_iter = iter(style_loader)
    for word in tqdm(my_words, desc="batch_number"):
        batch = next(loader_iter)
        style, laplace, wid = batch["style"][0], batch["laplace"][0], batch["wid"]

        # chop very long sequences to control width
        max_len = 128
        segments = [
            (style[i : i + max_len], laplace[i : i + max_len], wid[i : i + max_len])
            for i in range(0, len(style), max_len)
        ]

        for style_seg, laplace_seg, wid_seg in segments:
            style_t = style_seg.to(device)
            laplace_t = laplace_seg.to(device)
            text_ref = ContentData().get_content(word).to(device)
            text_ref = text_ref.repeat(style_t.size(0), 1, 1, 1)

            latent = torch.randn(
                (
                    text_ref.size(0),
                    4,
                    style_t.size(2) // 8,
                    (text_ref.size(1) * 32) // 8,
                ),
                device=device,
            )

            with torch.inference_mode(), autocast_ctx:
                if opt.sample_method == "ddim":
                    imgs = diffusion.ddim_sample(
                        unet,
                        vae,
                        style_t.size(0),
                        latent,
                        style_t,
                        laplace_t,
                        text_ref,
                        opt.sampling_timesteps,
                        opt.eta,
                    )
                elif opt.sample_method == "ddpm":
                    imgs = diffusion.ddpm_sample(
                        unet, vae, style_t.size(0), latent, style_t, laplace_t, text_ref
                    )
                else:
                    raise ValueError("sample_method should be 'ddim' or 'ddpm'")

            # save ------------------------------------------------------------
            for i, img in enumerate(imgs):
                pil = torchvision.transforms.ToPILImage()(img).convert("L")
                subdir = os.path.join(out_root, wid_seg[i][0])
                os.makedirs(subdir, exist_ok=True)
                pil.save(os.path.join(subdir, f"{word}.png"))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # keep the original flag name so main() can still access opt.cfg_file
    parser.add_argument("--cfg", dest="cfg_file", default="configs/IAM64.yml")
    parser.add_argument("--dir", dest="save_dir", default="Generated")
    parser.add_argument("--one_dm", required=True)
    parser.add_argument("--generate_type", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stable_dif_path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--sample_method", default="ddim")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    main(args)
