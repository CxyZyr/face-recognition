import torch
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from copy import deepcopy
from torch.utils.data import Dataset
from .utils import image_pipeline2, get_metrics
from tqdm import tqdm

class NDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode

        self.get_data()

    def get_data(self):
        self.data_items = []
        self.label_items = []

        with open(self.ann_path, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            path, label = line.rstrip().split()
            self.data_items.append(path)
            self.label_items.append(int(label))
        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))
        f.close()
        print('-----Load image and label is successful----- ')

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_items[idx]
        path = osp.join(self.data_dir, path)
        image = image_pipeline2(path, self.test_mode)
        label = self.label_items[idx]
        return image, label

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
