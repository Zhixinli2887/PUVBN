import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from glob import glob
import gc
from utils import *


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, tf=12, sf=12, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name
        self.tf = tf
        self.image_raw = cv2.imread(self.image_name)
        sh, sw, _ = self.image_raw.shape
        down_size = (int(sw / sf), int(sh / sf))
        self.image_raw = cv2.resize(self.image_raw, down_size)
        self.w, self.h = down_size
        self.image_raw = np.concatenate((self.image_raw, self.image_raw[:, 0:int(down_size[0] / 2), :]), axis=1)

        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)

    def __len__(self):
        return len(self.template_names)

    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        th, tw, _ = template.shape
        down_size = (int(tw / self.tf), int(th / self.tf))
        template = cv2.resize(template, down_size)
        if self.transform:
            template = self.transform(template)
        thresh = 0.7
        return {'image': self.image,
                'image_raw': self.image_raw,
                'image_name': self.image_name,
                'template': template.unsqueeze(0),
                'template_name': template_path,
                'template_h': template.size()[-2],
                'template_w': template.size()[-1],
                'thresh': thresh}


class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)

    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()

    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        if mode == 'big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]),
                                          mode='bilinear', align_corners=True)
        else:
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]),
                                          mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)


class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        eps = 1e-12
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / (x_std + eps)
        x2 = (x2 - x_mean) / (x_std + eps)
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]


class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None
    def __call__(self, template, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name is not image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class QATM():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()

        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

    def compute_output_shape(self, input_shape):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)