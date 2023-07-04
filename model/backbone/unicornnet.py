import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Optional

class Unicorn(nn.Module):
    def __init__(self,out_channel=256,fp16=False):
        super(Unicorn,self).__init__()

        self.out_channel = out_channel
        self.fp16 = fp16

        self.conv1a = nn.Conv2d(3, 16, 3, 1, 0)
        self.relu1a = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(16, 32, 3, 1, 0)
        self.relu1b = nn.ReLU(inplace=True)
        self.pool1b = nn.MaxPool2d(2,2)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_4 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.conv4_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_5 = nn.ReLU(inplace=True)
        self.conv4_6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_6 = nn.ReLU(inplace=True)
        self.conv4_7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_7 = nn.ReLU(inplace=True)
        self.conv4_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_8 = nn.ReLU(inplace=True)
        self.conv4_9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_9 = nn.ReLU(inplace=True)
        self.conv4_10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4_10 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.conv5_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_5 = nn.ReLU(inplace=True)
        self.conv5_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu5_6 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5_dw = nn.Conv2d(256,self.out_channel,4,1,0,groups=256)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.pool1b(self.relu1b(self.conv1b(self.relu1a(self.conv1a(x)))))
            short_cut = x
            x = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x))))
            x = short_cut + x

            x = self.pool2(self.relu2(self.conv2(x)))
            short_cut = x
            x = self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu3_4(self.conv3_4(self.relu3_3(self.conv3_3(x))))
            x = short_cut + x

            x = self.pool3(self.relu3(self.conv3(x)))
            short_cut = x
            x = self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu4_4(self.conv4_4(self.relu4_3(self.conv4_3(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu4_6(self.conv4_6(self.relu4_5(self.conv4_5(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu4_8(self.conv4_8(self.relu4_7(self.conv4_7(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu4_10(self.conv4_10(self.relu4_9(self.conv4_9(x))))
            x = short_cut + x

            x = self.pool4(self.relu4(self.conv4(x)))
            short_cut = x
            x = self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu5_4(self.conv5_4(self.relu5_3(self.conv5_3(x))))
            x = short_cut + x
            short_cut = x
            x = self.relu5_6(self.conv5_6(self.relu5_5(self.conv5_5(x))))
            x = short_cut + x

            x = self.relu5(self.conv5(x))
            x = self.conv5_dw(x)
            x = torch.flatten(x, 1)
        x = x.float() if self.fp16 else x
        return x

def _unicorn(arch,out_channel,fp16):
    model = Unicorn(out_channel=out_channel,fp16=fp16)

    return model

def unicorn_128(out_channel=128,fp16=False) -> Unicorn:
    return _unicorn('unicorn_128',out_channel=out_channel,fp16=fp16)

def unicorn_256(out_channel=256,fp16=True) -> Unicorn:
    return _unicorn('unicorn_256',out_channel=out_channel,fp16=fp16)

def unicorn_512(out_channel=512,fp16=False) -> Unicorn:
    return _unicorn('unicorn_512',out_channel=out_channel,fp16=fp16)



























