from __future__ import print_function
import os
import copy
import warnings
import sys

import torch
import torch.nn as nn

from utils import is_dist, get_world_size, get_rank
from importlib import import_module
from dataset.miniclass_dataset import MiniClassDataset
from dataset.pair_dataset import PairDataset
from dataset.n_dataset import NDataset

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

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


def build_from_cfg(cfg, module):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

    args = cfg.copy()

    obj_type = args.pop('type')
    if not isinstance(obj_type, str):
        raise TypeError(f'type must be a str, but got {type(obj_type)}')
    else:
        obj_cls = getattr(import_module(module), obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {module} module')

    return obj_cls(**args)


def build_train_dataloader(cfg):
    """
    Args:
        the type of `cfg` could also be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        PyTorch dataloader(s)
    """

    if 'dataset' not in cfg:
        raise KeyError(f'`cfg` must contain the key "dataset", but got {cfg}')
    dataset_type = cfg['dataset']['type']
    if dataset_type != 'MiniClassDataset':
        raise KeyError(f'train data must use MiniClassDataset now,but got {dataset_type}')

    data = MiniClassDataset(cfg['data_dir'],cfg['ann_path'],cfg['dataset']['test_mode'],cfg['dataset']['data_aug'])
    world_size = get_world_size()
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
                data, shuffle=cfg['dataloader']['shuffle'])
    else:
        sampler = None

    if 'dataloader' not in cfg:
        raise KeyError(f'`cfg` must contain the key "dataloader", but got {cfg}')
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=cfg['dataloader']['batch_size'],
                                             num_workers=cfg['dataloader']['num_workers'],
                                             pin_memory=cfg['dataloader']['pin_memory'],
                                             drop_last=cfg['dataloader']['drop_last'],
                                             sampler=sampler)
    return dataloader, sampler


def build_val_dataloader(cfg):
    """
    Args:
        the type of `cfg` could also be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        PyTorch dataloader(s)
    """
    if 'dataset' not in cfg:
        raise KeyError(f'`cfg` must contain the key "dataset", but got {cfg}')
    dataset_type = cfg['dataset']['type']
    if dataset_type != 'PairDataset':
        raise KeyError(f'train data must use PairDataset now,but got {dataset_type}')

    dataloader_all = []
    for database in cfg['database']:
        data = PairDataset(database['name'],database['data_dir'],database['ann_path'],cfg['dataset']['test_mode'])
        world_size = get_world_size()
        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                    data, shuffle=cfg['dataloader']['shuffle'])
        else:
            sampler = None

        if 'dataloader' not in cfg:
            raise KeyError(f'`cfg` must contain the key "dataloader", but got {cfg}')
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=cfg['dataloader']['batch_size'],
                                                 num_workers=cfg['dataloader']['num_workers'],
                                                 pin_memory=cfg['dataloader']['pin_memory'],
                                                 drop_last=cfg['dataloader']['drop_last'],
                                                 sampler=sampler)
        dataloader_all.append(dataloader)
    return dataloader_all


def build_n_val_dataloader(cfg):
    """
    Args:
        the type of `cfg` could also be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        PyTorch dataloader(s)
    """
    if 'dataset' not in cfg:
        raise KeyError(f'`cfg` must contain the key "dataset", but got {cfg}')
    dataset_type = cfg['dataset']['type']
    if dataset_type != 'NDataset':
        raise KeyError(f'train data must use NDataset now,but got {dataset_type}')

    # init val data
    val_dataloader_all = []
    for dataval in cfg['dataval']:
        valdata = NDataset(dataval['name'],dataval['data_dir'],dataval['ann_path'],cfg['dataset']['test_mode'])
        world_size = get_world_size()
        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                    valdata, shuffle=cfg['dataloader']['shuffle'])
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(valdata,
                                                 batch_size=cfg['dataloader']['batch_size'],
                                                 num_workers=cfg['dataloader']['num_workers'],
                                                 pin_memory=cfg['dataloader']['pin_memory'],
                                                 drop_last=cfg['dataloader']['drop_last'],
                                                 shuffle=False,
                                                 sampler=sampler)
        val_dataloader_all.append(dataloader)

    # init base data
    database = cfg['database']
    basedata = NDataset(database['name'], database['data_dir'], database['ann_path'], cfg['dataset']['test_mode'])
    world_size = get_world_size()
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            basedata, shuffle=cfg['dataloader']['shuffle'])
    else:
        sampler = None
    base_dataloader = torch.utils.data.DataLoader(basedata,
                                             batch_size=cfg['dataloader']['batch_size'],
                                             num_workers=cfg['dataloader']['num_workers'],
                                             pin_memory=cfg['dataloader']['pin_memory'],
                                             drop_last=cfg['dataloader']['drop_last'],
                                             shuffle=False,
                                             sampler=sampler)


    return base_dataloader,val_dataloader_all


def build_module(cfg, module):
    if 'net' not in cfg:
        raise KeyError(f'`cfg` must contain the key "net", but got {cfg}')
    rank = get_rank()
    net = build_from_cfg(cfg['net'], module)
    net = net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    if 'pretrained' in cfg:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pretrain_dict = torch.load(cfg['pretrained'],map_location=map_location)
        net.load_state_dict(pretrain_dict,strict=False)
        print('-----Load {} pretrain is Successful-----'.format(module))

    if 'clip_grad_norm' not in cfg:
        cfg['clip_grad_norm'] = 1e5
        warnings.warn('`clip_grad_norm` is not set. The default is 1e5')
    clip_grad_norm = cfg['clip_grad_norm']

    return {'net': net, 'clip_grad_norm': clip_grad_norm}


def build_model(cfg):
    if 'backbone' not in cfg:
        raise KeyError(f'`cfg` must contain the key "backbone", but got {cfg}')
    if 'head' not in cfg:
        raise KeyError(f'`cfg` must contain the key "head", but got {cfg}')

    model = {}
    for module in cfg:
        model[module] = build_module(cfg[module], f'model.{module}')
    return model

