# # models/recognition.py

import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import os
import copy

from .cnn_lstm import create_model

def partial_load_ocr_model(model, checkpoint_path):
    """
    Loads only the CNN layers from a checkpoint (skipping final conv or RNN weights)
    so you can partially initialize your OCR model.
    """
    print(f"Attempt partial load from {checkpoint_path} (OCR model - CNN only).")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    filtered = {}
    for k, v in state_dict.items():
        # Skip final conv6 and any RNN layers
        if "cnn.conv6" in k or "rnn." in k:
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Partial load done.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

def load_arabic_ocr_model(hw_weights_path: str, charset_path: str, partial: bool = False):
    """
    Builds a CRNN-based CNN-BLSTM model and loads the pretrained Arabic OCR weights.
    
    If partial==True, it loads only the CNN layers from the provided checkpoint (skipping
    final conv or RNN weights), so that you can integrate the finetuned last layer.
    """
    # 1) Load the character mapping
    with open(charset_path, 'r') as f:
        char_set = json.load(f)
    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    # 2) Prepare a config for your CRNN.
    #    Make sure num_of_outputs equals len(idx_to_char) + 1 (for the blank token).
    model_config = {
        "cnn_out_size": 1024,         # adjust as needed
        "num_of_channels": 3,         # or 3 or 4
        "num_of_outputs": len(idx_to_char) + 1,
        "use_instance_norm": False,   # adjust if your model was trained with instance norm
        "nh": 512                   # hidden size of LSTM
    }
    
    # 3) Create the model
    model = create_model(model_config)
    
    # 4) Load the pretrained weights
    if partial:
        partial_load_ocr_model(model, hw_weights_path)
    else:
        state_dict = torch.load(hw_weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded Arabic OCR weights from {hw_weights_path}")
    
    # 5) Return the model (and the character mapping)
    return model, idx_to_char

# def load_arabic_ocr_model(hw_weights_path: str, charset_path: str):
#     """
#     Builds a CRNN-based CNN-BLSTM model and loads the pretrained Arabic handwriting weights.
#     """
#     # 1) Load the character mapping
#     with open(charset_path, 'r') as f:
#         char_set = json.load(f)
#     # E.g. `char_set['char_to_idx']` and `char_set['idx_to_char']`
    
#     idx_to_char = {}
#     for k, v in char_set['idx_to_char'].items():
#         idx_to_char[int(k)] = v
    
#     # 2) Prepare a config for your CRNN
#     #    - Adjust these to match the exact channels/cnn_out_size that your hw.pt expects
#     model_config = {
#         "cnn_out_size": 1024,         # typical setting in your code
#         "num_of_channels": 3,        # or 3 or 4, depending on how you trained
#         "num_of_outputs": len(idx_to_char) + 1,
#         "use_instance_norm": False,   # or True if that's how you trained
#         "nh": 512                   # LSTM hidden size (so that 4 * 512 = 2048)
#     }
    
#     # 3) Create the model
#     model = create_model(model_config)
    
#     # 4) Load the pretrained weights
#     state_dict = torch.load(hw_weights_path, map_location='cpu')
#     model.load_state_dict(state_dict)
#     print(f"Loaded Arabic OCR weights from {hw_weights_path}")
    
#     # 5) Return the model (and anything else you need)
#     return model, idx_to_char
    
class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y

class VAE_CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(VAE_CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(4, 64, 3, [1, 1], 1),nn.BatchNorm2d(64),nn.ReLU(),
                                      nn.Conv2d(64, 128, 3, [1, 1], 1),nn.BatchNorm2d(128),nn.ReLU(),
                                      nn.Conv2d(128, 256, 3, [1, 1], 1),nn.BatchNorm2d(256),nn.ReLU()]
                                      )

    def forward(self, x):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y


class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
        #self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out