import torch
import torch.nn.functional as F
from torch import nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # 2-layer BiLSTM
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        # x: [T, B, nIn]
        recurrent, _ = self.rnn(x)  # => [T, B, 2*nHidden]
        T, B, H = recurrent.size()
        t_rec = recurrent.view(T * B, H)  # => [T*B, 2*nHidden]
        output = self.embedding(t_rec)    # => [T*B, nOut]
        output = output.view(T, B, -1)    # => [T, B, nOut]
        return output

class CRNN(nn.Module):
    """
    CNN + BiLSTM + log-softmax => shape [T, B, nclass].
    Matches a final conv layer of 512 channels (not 1024),
    and no BN at conv2, conv4, or conv6.
    """
    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False, use_instance_norm=False):
        super(CRNN, self).__init__()

        # final = 1024 at the 7th conv
        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel sizes
        ps = [1, 1, 1, 1, 1, 1, 0]  # paddings
        ss = [1, 1, 1, 1, 1, 1, 1]  # strides
        # We keep 512 at index 6 => [64, 128, 256, 256, 512, 512, 512]
        # No layer outputs 1024
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [64, 128, 256, 256, 512, 512, 1024]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            # i-th layer
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(nIn, nOut, kernel_size=ks[i], stride=ss[i], padding=ps[i])
            )
            # If checkpoint never used BN at these layers, skip batchNormalization
            # to match the keys in your hw.pt
            if batchNormalization:
                # If the checkpoint truly used BN, you'd set this to True at the relevant layers.
                if not use_instance_norm:
                    cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
                else:
                    cnn.add_module(f'instancenorm{i}', nn.InstanceNorm2d(nOut))

            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        # Build the 7 conv layers
        convRelu(0)                                # layer0
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)                                # layer1
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2, batchNormalization=False)       # layer2 (no BN)
        convRelu(3)                                # layer3
        cnn.add_module('pooling2',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, batchNormalization=False)       # layer4 (no BN)
        convRelu(5)
        cnn.add_module('pooling3',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, batchNormalization=False)       # layer6 => final channels=512

        self.cnn = cnn
        # Now connect to a BiLSTM with input dimension = cnnOutSize (512).
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax(dim=2)

    # def forward(self, x):
    #     # x: [B, C, H, W]
    #     conv = self.cnn(x)  # => [B, 512, 1, W] if everything is correct
    #     b, c, h, w = conv.size()

    #     # Flatten => [B, c*h, w] => [b, 512, w]
    #     conv = conv.view(b, c * h, w)
    #     # permute => [w, b, 512]
    #     conv = conv.permute(2, 0, 1)

    #     # LSTM => [T, B, nclass]
    #     output = self.rnn(conv)
    #     output = self.softmax(output)
    #     return output

    def forward(self, x):
        # x => shape [B, C, H_in, W_in]
        conv = self.cnn(x)  # => e.g. [B, 512, H_out, W_out]
        b, c, h, w = conv.shape

        # If your pretrained CRNN expects h=1, forcibly do it here:
        if h > 1:
            # forcibly collapse the height dimension to 1
            conv = F.adaptive_avg_pool2d(conv, (1, w))
            b, c, h, w = conv.shape  # now h=1

        # Now flatten => [B, c*h, w] => [B, 512, w]
        conv = conv.view(b, c * h, w)
        # permute => [w, B, 512]
        conv = conv.permute(2, 0, 1)
        # pass to LSTM => dimension matches
        output = self.rnn(conv)
        output = self.softmax(output)
        return output

# def create_model(config):
#     """
#     config keys:
#       - 'cnn_out_size': 512  (NOT 1024)
#       - 'num_of_channels': 3 (or 1)
#       - 'num_of_outputs': vocab size + 1
#       - 'use_instance_norm': bool
#     """
#     use_instance_norm = config.get('use_instance_norm', False)
#     cnn_out_size = config['cnn_out_size']         # e.g. 512
#     num_of_channels = config['num_of_channels']   # 3 or 1
#     num_of_outputs = config['num_of_outputs']     # vocab+1

#     crnn = CRNN(
#         cnnOutSize=cnn_out_size,
#         nc=num_of_channels,
#         nclass=num_of_outputs,
#         nh=512,  # LSTM hidden size
#         use_instance_norm=use_instance_norm
#     )
#     return crnn

def create_model(config):
    use_instance_norm = config.get('use_instance_norm', False)
    nh = config.get('nh', 512)  # Use the hidden size from config, default to 512 if not provided
    crnn = CRNN(
        config['cnn_out_size'], 
        config['num_of_channels'], 
        config['num_of_outputs'], 
        nh, 
        use_instance_norm=use_instance_norm
    )
    return crnn

