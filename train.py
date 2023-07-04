import os
import yaml
import argparse

import torch
import torch.distributed as dist

from runner import IterRunner


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config',default='./config/train/unicorn256.yml',
            help='train config file path')
    parser.add_argument('--local_rank',default=0,
            help='rank')
    args = parser.parse_args()

    return args

def main_worker(config):
    # init processes
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    #init runner and run
    runner = IterRunner(config)
    runner.train()

    # clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config,'r') as f:
        config = yaml.load(f,yaml.SafeLoader)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.''CPU mode is not supported yet')

    main_worker(config)


