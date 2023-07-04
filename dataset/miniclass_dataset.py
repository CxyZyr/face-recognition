from __future__ import print_function
import random
import os.path as osp
import sys
import numpy as np
from copy import deepcopy

from .utils import image_pipeline2
from torch.utils.data import Dataset

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from tqdm import tqdm

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


class MiniClassDataset(Dataset):
    def __init__(self, data_dir, ann_path, test_mode=False,data_aug=False):
        super().__init__()

        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode
        self.data_aug = data_aug

        self.get_data()

    def get_data(self):
        """Get data from a provided annotation file.
        """
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
        print('-----Load image_path and label is successful----- ')

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_items[idx]
        path = osp.join(self.data_dir,path)
        image = image_pipeline2(path, self.test_mode,self.data_aug)
        label = self.label_items[idx]
        # image = deepcopy(image)
        # label = deepcopy(label)
        return image, label

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
