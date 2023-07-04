# -*- coding:utf-8 -*-
import os
import torch
import torch.onnx
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from model.backbone.unicornnet import unicorn_256
import onnx
from onnxsim import simplify

def save_model(pth_file):
    net = unicorn_256(fp16=False)
    net_dict = torch.load(pth_file,map_location=torch.device('cpu'))
    net_dict_2 = {key.replace('module.',''):value for key,value in net_dict.items()}
    net.load_state_dict(net_dict_2)
    torch.save(net,'./UnicornNet_Mask.pt')

def onnx_export(pth_file):
    net = unicorn_256(fp16=False)
    net_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    net_dict_2 = {key.replace('module.', ''): value for key, value in net_dict.items()}
    net.load_state_dict(net_dict_2)
    net.eval()
    dummpy_input = torch.zeros(1, 3, 128, 128)
    onnx_name = r'UnicornNet_Mask.onnx'
    torch.onnx.export(
        net, dummpy_input, onnx_name,
        verbose=True,
        input_names=['image'],
        output_names=['predict'],
        opset_version=11,
        dynamic_axes=None  # 注意这里指定batchsize是动态可变的
    )

def onnx_sim(onnx_path):
    model_onnx = onnx.load_model(onnx_path)
    model_smi, check = simplify(model_onnx)
    save_path = 'simple_UnicornNet_Mask.onnx'
    onnx.save(model_smi, save_path)
    print('模型静态图简化完成')

if __name__ == '__main__':
    pth_file = r'./weights/0522_backbone_25000.pth'
    onnx_path = r'UnicornNet_Mask.onnx'
    # save_model(pth_file)
    onnx_export(pth_file)
    onnx_sim(onnx_path)


