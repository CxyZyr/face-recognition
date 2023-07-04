import os
import os.path as osp
import time
import yaml
import warnings

import torch
import torch.optim as optim

from utils import get_world_size, get_rank
from builder import build_train_dataloader, build_val_dataloader,build_model
from utils import Logger,CosineDecayLR

from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class IterRunner():
    def __init__(self, config):
        self.config = config
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.iter = 0

        # init dataloader
        self.train_dataloader,self.sampler = build_train_dataloader(self.config['train']['data'])
        self.val_dataloader = build_val_dataloader(self.config['val'])

        # init model
        feat_dim = config['model']['backbone']['net']['out_channel']
        self.config['model']['head']['net']['feat_dim'] = feat_dim
        self.model = build_model(config['model'])

        # init project
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.project_dir = osp.join(config['common']['save_log_dir'],timestamp)
        os.makedirs(self.project_dir,exist_ok=True)
        if self.rank == 0:
            print('')
            print('The training log and models are saved to ' + self.project_dir)
            print('')

        # save cfg
        save_cfg_path = osp.join(self.project_dir,config['common']['save_cfg_name'])
        with open(save_cfg_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)

        # save log
        save_log_dir = osp.join(self.project_dir, 'log')
        os.makedirs(save_log_dir, exist_ok=True)
        self.train_log = Logger(name='train', path="{}/{}_train.log".format(save_log_dir,time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
        self.val_log = Logger(name='val', path="{}/{}_val.log".format(save_log_dir,time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))

        #save weight
        self.save_weights_dir = osp.join(self.project_dir,'weights')
        os.makedirs(self.save_weights_dir, exist_ok=True)

        # init common and train arguments
        self.freeze_epoch = self.config['train']['freeze']['epoch']
        self.norm_epoch = self.config['train']['norm']['epoch']
        self.test_first = self.config['common']['test_first']
        self.screen_intvl = self.config['common']['screen_intvl']
        self.val_intvl = self.config['common']['val_intvl']
        self.save_iters = self.config['common']['save_iters']
        self.freeze_iter_step = self.config['train']['freeze']['optim']['iter_step']
        self.norm_iter_step = self.config['train']['norm']['optim']['iter_step']
        self.scheduler_type = None
        self.tpr_1e_3 = 0
        self.tpr_5e_3 = 0
        self.acc = 0

        # make sure the max_save_iter less than all_iter
        all_iter = (self.freeze_epoch+self.norm_epoch)*len(self.train_dataloader)
        if self.rank == 0:
            if len(self.save_iters) == 0:
                warnings.warn('`save_iters` is not set. if you want to save model in specified location,or not only end of each epoch.please check it!')
            else:
                if all_iter < max(self.save_iters):
                    raise KeyError(f'all_iter is {all_iter},but got max_save_iter {max(self.save_iters)},max_save_iter must be less than it')

        if self.rank != 0:
            return

    def set_optimizer_scheduler(self,config,freeze=False):
        for module in self.model:
            if freeze:
                for param in self.model['backbone']['net'].parameters():
                    param.requires_grad = False
            else:
                for param in self.model['backbone']['net'].parameters():
                    param.requires_grad = True
            self.model[module]['optimizer'] = optim.SGD(self.model[module]['net'].parameters(),
                          lr=config['optim']['lr_init'],
                          momentum=config['optim']['momentum'],
                          weight_decay=config['optim']['weight_decay'])
            if config['scheduler']['type'] == 'CosineDecayLR':
                self.scheduler_type = 'CosineDecayLR'
                self.model[module]['scheduler'] = CosineDecayLR(
                    self.model[module]['optimizer'],
                    T_max=config['epoch']*len(self.train_dataloader),
                    lr_init=config['optim']['lr_init'],
                    lr_min=config['scheduler']['lr_end'],
                    warmup=config['scheduler']['warm_up_epoch']*len(self.train_dataloader)
                )
            if config['scheduler']['type'] == 'MultiStepLR':
                self.scheduler_type = 'MultiStepLR'
                self.model[module]['scheduler'] = optim.lr_scheduler.MultiStepLR(
                    self.model[module]['optimizer'],
                    config['scheduler']['milestones'],
                    config['scheduler']['gamma'],
                    -1
                )
    def set_model(self, test_mode):
        for module in self.model:
            if test_mode:
                self.model[module]['net'].eval()
            else:
                self.model[module]['net'].train()

    def update_model(self,i,freeze=False):
        for module in self.model:
            if freeze:
                if i % self.freeze_iter_step == 0:
                    self.model[module]['optimizer'].step()
                    self.model[module]['optimizer'].zero_grad()
                    if self.scheduler_type == 'CosineDecayLR':
                        self.model[module]['scheduler'].step(self.iter)
                    else:
                        self.model[module]['scheduler'].step()
            else:
                if i % self.norm_iter_step == 0:
                    self.model[module]['optimizer'].step()
                    self.model[module]['optimizer'].zero_grad()
                    if self.scheduler_type == 'CosineDecayLR':
                        self.model[module]['scheduler'].step(self.iter-self.freeze_epoch*len(self.train_dataloader))
                    else:
                        self.model[module]['scheduler'].step()


    def save_model(self):
        for module in self.model:
            model_name = '{}_{}.pth'.format(str(module), str(self.iter+1))
            model_path = osp.join(self.save_weights_dir, model_name)
            torch.save(self.model[module]['net'].state_dict(), model_path)

    @torch.no_grad()
    def val(self):
        # switch to test mode
        self.set_model(test_mode=True)
        for val_loader in self.val_dataloader:
            # meta info
            dataset = val_loader.dataset
            # create a placeholder `feats`,
            # compute _feats in different GPUs and collect
            dim = self.config['model']['backbone']['net']['out_channel']
            with torch.no_grad():
                feats = torch.zeros(
                    [len(dataset), dim], dtype=torch.float32).to(self.rank)
                for data, indices in val_loader:
                    data = data.to(self.rank)
                    _feats = self.model['backbone']['net'](data)
                    data = torch.flip(data, [3])
                    _feats += self.model['backbone']['net'](data)
                    feats[indices, :] = _feats

                dist.all_reduce(feats, op=dist.ReduceOp.SUM)
                results = dataset.evaluate(feats.cpu())
            if self.rank == 0:
                results = dict(results)
                self.val_log.logger.info("Processing Val Iter:{} [{} : {}]".format(self.iter+1, dataset.name, results))
                # if model have acc better in the test data,save the model
                if results['TPR@FPR=1e-3'] >= self.tpr_1e_3 or results['ACC'] >= self.acc:
                    self.save_model()
                    self.tpr_1e_3 = results['TPR@FPR=1e-3']
                    self.acc = results['ACC']


    def train(self):
        if self.test_first:
            self.val()
        self.set_optimizer_scheduler(self.config['train']['freeze'],freeze=True)
        for epoch in range(self.freeze_epoch):
            Loss,Mag_mean,Mag_std,bkb_grad,head_grad = 0,0,0,0,0
            if self.sampler != None:
                self.sampler.set_epoch(epoch)
            self.set_model(test_mode=False)
            for i,(images,labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.rank), labels.to(self.rank)

                # forward
                self.set_model(test_mode=False)
                feats = self.model['backbone']['net'](images)
                loss = self.model['head']['net'](feats, labels)

                # backward
                loss.backward()
                b_norm = self.model['backbone']['clip_grad_norm']
                h_norm = self.model['head']['clip_grad_norm']
                if b_norm < 0. or h_norm < 0.:
                    raise ValueError(
                            'the clip_grad_norm should be positive. ({:3.4f}, {:3.4f})'.format(b_norm, h_norm))

                b_grad = clip_grad_norm_(
                        self.model['backbone']['net'].parameters(),
                        max_norm=b_norm, norm_type=2)
                h_grad = clip_grad_norm_(
                        self.model['head']['net'].parameters(),
                        max_norm=h_norm, norm_type=2)

                # update model
                self.iter = epoch*len(self.train_dataloader)+i
                self.update_model(i,freeze=True)

                magnitude = torch.norm(feats, 2, 1)
                Loss = (Loss * i + loss.item()) / (i + 1)
                Mag_mean = (Mag_mean * i + magnitude.mean().item()) / (i + 1)
                Mag_std = (Mag_std * i + magnitude.std().item()) / (i + 1)
                bkb_grad = (bkb_grad * i + b_grad) / (i + 1)
                head_grad = (head_grad * i + h_grad) / (i + 1)

                if (i + 1) % self.screen_intvl == 0 or (i + 1) == len(self.train_dataloader):
                    if self.rank == 0:
                        # logging and update meters
                        self.train_log.logger.info("Processing Freeze Training Epoch:[{} | {}] Batch:[{} | {}] Lr:{:.6f} Loss:{:.4f} Mag_mean:{:.4f} Mag_std:{:.4f} bkb_grad:{:.4f} head_grad:{:.4f}"
                             .format(epoch+1,self.freeze_epoch+self.norm_epoch,i+1,len(self.train_dataloader),self.model['backbone']['optimizer'].param_groups[0]['lr'],Loss, Mag_mean, Mag_std, bkb_grad, head_grad))

                # if (i + 1) % self.val_intvl == 0 or (i + 1) == len(self.train_dataloader) or (self.iter + 1) in self.save_iters:
                #     self.val()
                if ((self.iter + 1) in self.save_iters or (i + 1) == len(self.train_dataloader)) and self.rank == 0:
                    self.save_model()

        self.set_optimizer_scheduler(self.config['train']['norm'], freeze=False)

        for epoch in range(self.norm_epoch):
            Loss,Mag_mean,Mag_std,bkb_grad,head_grad = 0,0,0,0,0
            if self.sampler != None:
                self.sampler.set_epoch(epoch)
            self.set_model(test_mode=False)
            for i,(images,labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.rank), labels.to(self.rank)

                # forward
                self.set_model(test_mode=False)
                feats = self.model['backbone']['net'](images)
                loss = self.model['head']['net'](feats, labels)

                # backward
                loss.backward()
                b_norm = self.model['backbone']['clip_grad_norm']
                h_norm = self.model['head']['clip_grad_norm']
                if b_norm < 0. or h_norm < 0.:
                    raise ValueError(
                            'the clip_grad_norm should be positive. ({:3.4f}, {:3.4f})'.format(b_norm, h_norm))

                b_grad = clip_grad_norm_(
                        self.model['backbone']['net'].parameters(),
                        max_norm=b_norm, norm_type=2)
                h_grad = clip_grad_norm_(
                        self.model['head']['net'].parameters(),
                        max_norm=h_norm, norm_type=2)

                # update model
                self.iter = (self.freeze_epoch+epoch)*len(self.train_dataloader)+i
                self.update_model(i,freeze=False)

                magnitude = torch.norm(feats, 2, 1)
                Loss = (Loss * i + loss.item()) / (i + 1)
                Mag_mean = (Mag_mean * i + magnitude.mean().item()) / (i + 1)
                Mag_std = (Mag_std * i + magnitude.std().item()) / (i + 1)
                bkb_grad = (bkb_grad * i + b_grad) / (i + 1)
                head_grad = (head_grad * i + h_grad) / (i + 1)

                if (i + 1) % self.screen_intvl == 0 or (i + 1) == len(self.train_dataloader):
                    if self.rank == 0:
                        # logging and update meters
                        self.train_log.logger.info("Processing Norm Training Epoch:[{} | {}] Batch:[{} | {}] Lr:{:.6f} Loss:{:.4f} Mag_mean:{:.4f} Mag_std:{:.4f} bkb_grad:{:.4f} head_grad:{:.4f}"
                            .format(epoch+self.freeze_epoch+1, self.freeze_epoch + self.norm_epoch, i+1,len(self.train_dataloader),self.model['backbone']['optimizer'].param_groups[0]['lr'],Loss, Mag_mean, Mag_std, bkb_grad, head_grad))
                # do test
                if (i + 1) % self.val_intvl == 0 or (i + 1) == len(self.train_dataloader) or (self.iter + 1) in self.save_iters:
                    self.val()
                # do save
                if ((self.iter + 1) in self.save_iters or (i + 1) == len(self.train_dataloader)) and self.rank == 0:
                    self.save_model()







