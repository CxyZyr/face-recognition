import os
import os.path as osp
import yaml
import time
import argparse

import torch
import torch.nn as nn

from tabulate import tabulate
from datetime import datetime

from builder import build_val_dataloader, build_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='A PyTorch project for face recognition.')
    parser.add_argument('--config-path',default='./config/val/val_unicorn256.yml',
                        help='config files for testing datasets')
    args = parser.parse_args()

    return args


@torch.no_grad()
def get_feats(net, data, flip=True):
    # extract features from the original
    # and horizontally flipped data
    with torch.no_grad():
        feats = net(data)
        # if flip:
        #     data = torch.flip(data, [3])
        #     feats += net(data)
    return feats.data.cpu()


@torch.no_grad()
def test_run(model_dir,net, checkpoints, dataloaders):
    # init save path
    save_val_dir = 'val/val_{}.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()))
    os.makedirs(os.path.dirname(save_val_dir), exist_ok=True)
    with open(save_val_dir, 'a', encoding='utf-8') as f:
        f.write('{}:\n'.format(model_dir))

        print('Val Start:')
        for n_ckpt, checkpoint in enumerate(checkpoints):
            f.write('{}:\n'.format(checkpoint))
            # load model parameters
            net.load_state_dict(torch.load(checkpoint))
            for n_loader, dataloader in enumerate(dataloaders):
                # get feats from test_loader
                dataset_feats = []
                dataset_indices = []
                for n_batch, (data, indices) in enumerate(dataloader):
                    # collect feature and indices
                    data = data.cuda()
                    indices = indices.tolist()
                    # with torch.no_grad():
                    feats = get_feats(net, data)

                    dataset_feats.append(feats)
                    dataset_indices.extend(indices)

                # eval
                dataset_feats = torch.cat(dataset_feats, dim=0)
                dataset_feats = dataset_feats[dataset_indices]
                results = dataloader.dataset.evaluate(dataset_feats)
                results = dict(results)
                name = dataloader.dataset.name

                f.write('   {} : {}\n'.format(name,results))
        print('Val Finish')



def main_worker(test_config):
    # parallel setting
    device_ids = os.environ['CUDA_VISIBLE_DEVICES']
    device_ids = [int(id) for id in device_ids.split(',')]
    device_ids = list(range(len(device_ids)))

    # build model
    bkb_net = build_from_cfg(
        test_config['model']['backbone']['net'],
        'model.backbone',
    )

    bkb_net = nn.DataParallel(bkb_net, device_ids=device_ids)
    bkb_net = bkb_net.cuda()
    bkb_net.eval()

    # model paths and run test
    model_dir = test_config['project']['model_dir']
    model_list = test_config['project']['model_list']
    bkb_paths = [os.path.join(model_dir,backbone) for backbone in model_list]

    # build dataloader
    test_loaders = build_val_dataloader(test_config['data'])

    # evaluate each dataset
    test_run(model_dir,bkb_net, bkb_paths, test_loaders)



if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                       'CPU mode is not supported yet')

    main_worker(config)
