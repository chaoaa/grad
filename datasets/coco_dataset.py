import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle

import json
import cv2
from PIL import Image

# 图像读取预处理单元
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp

class CocoDataset(data.Dataset):
    def __init__(
        self,
        image_ids_path,
        input_seq,
        target_seq,
        gv_feat_path,
        att_feats_folder,
        seq_per_img,
        max_feat_num,
        keyword_mask_path=None,   # KSC: 关键词 mask 文件路径，为空则不加载
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        # 此处 image_ids_path 为 ids2path 的映射 dict
        with open(image_ids_path, 'r') as f:
            self.ids2path = json.load(f)           # dict {image_id: image_path}
            self.image_ids = list(self.ids2path.keys())  # list of str

        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None

        # 构建图像预处理单元
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=str_to_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )

        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0, :])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None

        # KSC: 加载关键词 mask（可选）
        if keyword_mask_path and len(keyword_mask_path) > 0:
            self.keyword_mask = pickle.load(open(keyword_mask_path, 'rb'), encoding='bytes')
        else:
            self.keyword_mask = None
         
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        ########################################-------online test(需onlineTest则注释掉此部分，反之则相反操作)------############################
        image_path = self.ids2path[image_id]
        ########################################------------------------############################
        indices = np.array([index]).astype('int')

        if self.gv_feat is not None:
            gv_feat = self.gv_feat[image_id]
            gv_feat = np.array(gv_feat).astype('float32')
        else:
            gv_feat = np.zeros((1,1))

        # 此处att_feats_folder为coco数据集源图像保存路径，而非预训练特征保存路径
        if self.att_feats_folder is not None:
             att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['feat']
             att_feats = np.array(att_feats).astype('float32')
            # 读取图像，并进行预处理
            # image_path = self.ids2path[image_id]
            # img = cv2.imread(os.path.join(self.att_feats_folder, image_path))
            # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # att_feats = self.transform(img)  # [3, 384, 384]，图像
            
        else:
            # att_feats = np.zeros((1,1))
            att_feats = torch.zeros(1, 1)
        
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
           att_feats = att_feats[:self.max_feat_num, :]

        if self.seq_len < 0:
            # 验证/测试阶段：保持原始返回，不含 keyword_seq
            return indices, gv_feat, att_feats

        input_seq  = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        # keyword_seq 与 target_seq 形状完全对齐，默认全 0（无关键词）
        keyword_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int32')

        n = len(self.input_seq[image_id])
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :]   = self.input_seq[image_id]
            target_seq[0:n, :]  = self.target_seq[image_id]
            # keyword_mask 与 target 同步填充
            if self.keyword_mask is not None:
                km = self.keyword_mask.get(image_id, None)
                if km is not None:
                    km = np.array(km, dtype='int32')
                    # 对齐 seq_len（keyword_mask 可能比 seq_len 长或短）
                    copy_len = min(km.shape[1], self.seq_len) if km.ndim == 2 else min(len(km[0]), self.seq_len)
                    keyword_seq[0:n, :copy_len] = km[0:n, :copy_len]

        for i, ix in enumerate(ixs):
            input_seq[sid + i]  = self.input_seq[image_id][ix, :]
            target_seq[sid + i] = self.target_seq[image_id][ix, :]
            if self.keyword_mask is not None:
                km = self.keyword_mask.get(image_id, None)
                if km is not None:
                    km = np.array(km, dtype='int32')
                    copy_len = min(km.shape[1], self.seq_len) if km.ndim == 2 else self.seq_len
                    if ix < km.shape[0]:
                        keyword_seq[sid + i, :copy_len] = km[ix, :copy_len]

        return indices, input_seq, target_seq, keyword_seq, gv_feat, att_feats
