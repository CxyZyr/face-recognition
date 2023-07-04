import os
import os.path as osp
import yaml
import time
import argparse
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabulate import tabulate
from datetime import datetime
from tqdm import tqdm

from builder import build_from_cfg,build_n_val_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description='A PyTorch project for face recognition.')
    parser.add_argument('--config-path',default='./config/val/val_sfz_10w_1_n.yml',
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
def test_run(model_dir, net, checkpoints, base_loaders, val_loaders):
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

            # get basedata feats
            base_feats = []
            base_indices = []
            for n_batch,(data,indices) in enumerate(base_loaders):
                data = data.cuda()
                indices = indices.tolist()

                feats = get_feats(net, data)
                base_feats.append(feats)
                base_indices.extend(indices)

            base_feats = torch.cat(base_feats, dim=0)
            base_feats = F.normalize(base_feats, dim=1)
            # get val feats
            for n_loader, dataloader in enumerate(val_loaders):
                # get feats from val_loader
                val_feats = []
                val_indices = []
                for n_batch, (data, indices) in enumerate(dataloader):
                    # collect feature and indices
                    data = data.cuda()
                    indices = indices.tolist()
                    feats = get_feats(net, data)

                    val_feats.append(feats)
                    val_indices.extend(indices)
                val_feats = torch.cat(val_feats, dim=0)
                # eval
                name = dataloader.dataset.name
                print('{}/{} match start:'.format(checkpoint,name))
                top_1,top_5,top_10 = 0,0,0
                match_scores = []
                for i in tqdm(range(len(val_feats))):
                    # get cosine scores
                    val_feat = torch.zeros_like(base_feats)
                    val_feat[:] = val_feats[i]
                    val_feat = F.normalize(val_feat, dim=1)
                    scores = torch.sum(base_feats * val_feat, dim=1).tolist()
                    # get match inindices
                    max_indices = heapq.nlargest(10, range(len(scores)), key=scores.__getitem__)
                    match_indices = [base_indices[i] for i in max_indices]
                    # save match scores
                    save_line = ''
                    for j in max_indices:
                        save_line = save_line + '{} '.format(scores[j])
                    save_line = save_line + '\n'
                    match_scores.append(save_line)
                    # get Top_K_Accuracy
                    if val_indices[i] in match_indices[:1]:
                        top_1 += 1
                    if val_indices[i] in match_indices[:5]:
                        top_5 += 1
                    if val_indices[i] in match_indices:
                        top_10 += 1
                scores_txt = '{}_{}.txt'.format(checkpoint, name)
                with open(scores_txt, 'a', encoding='utf-8') as f1:
                    f1.writelines(match_scores)
                results = {'Top_1_Accuracy': top_1/len(val_feats), 'Top_5_Accuracy': top_5/len(val_feats), 'Top_10_Accuracy': top_10/len(val_feats)}
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
    base_loaders,val_loaders= build_n_val_dataloader(test_config['data'])

    # evaluate each dataset
    test_run(model_dir, bkb_net, bkb_paths, base_loaders, val_loaders)



if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                       'CPU mode is not supported yet')

    main_worker(config)
