import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader_ara import IAMDataset
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.recognition import HTRNet
# from models.recognition import load_arabic_ocr_model
from data_loader.loader_ara import letters
from models.loss import SupConLoss

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)
    
    """ set dataset"""
    train_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE)
    print('number of training images: ', len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    
    val_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.VAL.TYPE)
    val_sampler = DistributedSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=cfg.TEST.IMS_PER_BATCH,
                                              drop_last=False,
                                              collate_fn=val_dataset.collate_fn_,
                                              pin_memory=True,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                              sampler=val_sampler)
    
    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    
    """load pretrained model"""
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        print('no pretrained one_dm model loaded')
        exit()

    unet = DDP(unet, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    ctc_loss = nn.CTCLoss()
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    '''load pretrained ocr model'''
    # ocr_model = HTRNet(nclasses = len(letters), vae=True)
    
    # if len(opt.ocr_model) > 0:
    #     # miss, unxep = ocr_model.load_state_dict(torch.load(opt.ocr_model, map_location=torch.device('cpu')), strict=False)
    #     # ocr_model, idx_to_char = load_arabic_ocr_model(opt.ocr_model, "models/charset.json")

    #     # To load the OCR model with partial loading:
    #     ocr_model, idx_to_char = load_arabic_ocr_model("./ocr_checkpoints/ocr_best_state.pth", "models/charset.json", partial=True)

    #     ocr_model = ocr_model.to(device)
    #     ocr_model.requires_grad_(False)  # freeze it so it doesn't get updated
    #     # print('load pretrained ocr model from {}'.format(opt.ocr_model))
    #     print('Loaded the Arabic OCR model.')
    # else:
    #     print('failed to load the pretrained ocr model')
    #     exit()
    # # ocr_model.requires_grad_(False)
    # # ocr_model = ocr_model.to(device)
    
    '''load pretrained ocr model'''
    ocr_model = HTRNet(nclasses = len(letters)+1, vae=False)
    if len(opt.ocr_model) > 0:
        # miss, unxep = ocr_model.load_state_dict(torch.load(opt.ocr_model, map_location=torch.device('cpu')), strict=False)
        ocr_model.load_state_dict(torch.load(opt.ocr_model, map_location='cpu'), strict=True)
        print('load pretrained ocr model from {}'.format(opt.ocr_model))
    else:
        print('failed to load the pretrained ocr model')
        exit()
    ocr_model.requires_grad_(False)
    ocr_model = ocr_model.to(device)
    
    
    """load pretrained vae"""
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae = vae.to(device)


    """build trainer"""
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, val_loader, device, ocr_model, ctc_loss)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64_finetune.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--ocr_model', dest='ocr_model', default='./model_zoo/vae_HTR138.pth', help='pre-trained ocr model')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)