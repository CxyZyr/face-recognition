import torch
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.data import Dataset
from .utils import image_pipeline, get_metrics


class PairDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from an annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        paths = set()
        for line in lines:
            _, path1, path2 = line.rstrip().split(' ')
            paths.add(path1)
            paths.add(path2)
        paths = list(paths)
        paths.sort()
        self.data_items = [{'path': path} for path in paths]

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def get_label(self):
        """Get labels from an annoation file
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        path2index = {item['path']: idx 
                for idx, item in enumerate(self.data_items)}

        self.indices0 = []
        self.indices1 = []
        self.labels = []
        for line in lines:
            label, path0, path1 = line.rstrip().split(' ')
            self.indices0.append(path2index[path0])
            self.indices1.append(path2index[path1])
            self.labels.append(int(label))

    def prepare(self, idx):
        # load image and pre-process (pipeline) from path
        path = self.data_items[idx]['path']
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)
        # image = deepcopy(image)

        return image, idx

    def tpr_at_fpr(self,y_true, y_scores, fpr_threshold=0.01):
        """
        计算给定FPR阈值下的TPR@FPR值。
        参数：
            y_true - 一维数组，表示真实标签（0或1）
            y_scores - 一维数组，表示预测标签的置信度或概率值
            fpr_threshold - FPR的阈值，默认为0.01
        返回值：
            TPR@FPR - 浮点数，表示在给定FPR阈值下的TPR@FPR值
        """
        # 将y_scores按照降序排序
        indices = np.argsort(-y_scores)
        y_true_sorted = y_true[indices]
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive

        # 计算给定FPR阈值下的TPR@FPR值
        n_tp = 0
        n_fp = 0
        threshold = 0
        for i in range(len(y_true)):
            if y_true_sorted[i] == 1:
                n_tp += 1
            else:
                n_fp += 1
            fpr = n_fp / n_negative
            if fpr >= float(fpr_threshold):
                y_scores = -y_scores
                y_scores = y_scores[indices]
                threshold = -y_scores[i]
                break
        tpr = n_tp / n_positive

        return tpr, threshold

    def evaluate(self, feats, FPRs=['1e-4','5e-4','1e-3','5e-3','5e-2']):
        # pair-wise scores
        feats = F.normalize(feats, dim=1)
        feats0 = feats[self.indices0, :]
        feats1 = feats[self.indices1, :]
        scores = torch.sum(feats0 * feats1, dim=1).tolist()

        # save_lines = []
        # for score in scores:
        #     save_line = '{}\n'.format(str(score))
        #     save_lines.append(save_line)
        # with open('result.txt', 'a', encoding='utf-8') as f:
        #     f.writelines(save_lines)
        # result = {}
        # labels = np.array(self.labels)
        # if isinstance(FPRs, list):
        #     for FPR in FPRs:
        #         tpr, threshold = self.tpr_at_fpr(labels, scores, fpr_threshold=FPR)
        #         # print('TPR@FPR={} : {} Threshold : {}'.format(FPR,tpr,threshold))
        #         key, value = 'TPR@FPR={}'.format(FPR), tpr
        #         result[key] = value
        #
        # else:
        #     tpr, threshold = self.tpr_at_fpr(labels, scores, fpr_threshold=FPRs)
        #     # print('TPR@FPR={} : {} Threshold : {}'.format(FPRs, tpr, threshold))
        #     key, value = 'TPR@FPR={}'.format(FPRs), tpr
        #     result[key] = value

        return get_metrics(self.labels, scores, FPRs)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
