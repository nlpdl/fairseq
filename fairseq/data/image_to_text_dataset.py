
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import io
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import math
from fairseq.data import FairseqDataset,data_utils
from fairseq.data.image_utils import (
    parse_path,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
import cv2


logger = logging.getLogger(__name__)


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

class RawImageDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path,
        img_path,
        src,
        split,
        min_sample_size=0,
        shuffle=True,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.text_compressor = TextCompressor(level=text_compression_level)
        skipped = 0
        self.fnames = []#图片
        self.split = split
        # self.white_fnames = []#白背景
        # self.mask_fnames = []#去噪背景
        sizes = []
        self.transform = NormalizePAD((3, 32, 100))
        self.skipped_indices = set()
        self.root_dir = manifest_path
        self.img_path = img_path
        # with open(manifest_path+'/{}_img.{}'.format(split,src), "r") as f:#读取图像
            # self.root_dir = f.readline().strip()
        with open(manifest_path+'/{}.img.{}'.format(split,src), "r") as f:#读取图像
            for i, line in enumerate(f):
                # items = line.strip()#分为图像地址与图像大小，图像大小统一640**480
                items = line.strip().split('\t')[0]
                # assert len(items) == 2, line
                # sz = int(items[1])
                # if min_sample_size is not None and sz < min_sample_size:
                #     skipped += 1
                #     self.skipped_indices.add(i)
                #     continue
                self.fnames.append(self.text_compressor.compress(items))
                # sizes.append(sz)
        # logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")


        # with open(manifest_path+'/{}_mask.{}'.format(split,src), "r") as f:#读取图像
        #     for i, line in enumerate(f):
        #         items = line.strip()#分为图像地址与图像大小，图像大小统一640**480
        #         self.mask_fnames.append(self.text_compressor.compress(items))
        
        # with open(manifest_path+'/{}_white.{}'.format(split,src), "r") as f:#读取图像
        #     for i, line in enumerate(f):
        #         items = line.strip()#分为图像地址与图像大小，图像大小统一640**480
        #         self.white_fnames.append(self.text_compressor.compress(items))

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
            # self.white_fnames = pyarrow.array(self.white_fnames)
            # self.mask_fnames = pyarrow.array(self.mask_fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass
    def feat(self,_path):
        img = cv2.imread(_path,cv2.IMREAD_GRAYSCALE)#这里用灰度图

        img = Image.fromarray(img, 'L')
        w, h = img.size
        img = np.array(img.convert("L"))
        img = adjust_contrast_grey(img, target = 0.5)
        img = Image.fromarray(img, 'L')
        ratio = w / float(h)
        if math.ceil(32 * ratio) > 100:
            resized_w = 100
        else:
            resized_w = math.ceil(32 * ratio)
        resized_image = img.resize((resized_w, 32), Image.BICUBIC)
        feats = self.transform(resized_image)
        return feats
    def padding_feat(self,img):
        s_h,s_w = 680,480
        h, w, n = img.shape
        if w > 480:
            img = cv2.resize(img, (480, int(h*(float(480)/float(w)))))
            h, w, n = img.shape
        img_modify = cv2.copyMakeBorder(img, 0, s_h - h, 0, s_w - w, cv2.BORDER_CONSTANT, value=[0,0,0,0])
        return img_modify


    def __getitem__(self, index):
        
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        
        path_or_fp = os.path.join(self.img_path, fn)
        # path_or_fp = os.path.join('/home/sxy/Projects/cp/base_data/img_data/', fn)
        # path_or_fp = os.path.join(self.root_dir+'/img_crop', fn)
        # print(path_or_fp)
        _path = parse_path(path_or_fp)
        img = cv2.imread(_path)
        img = self.padding_feat(img)
        feats = torch.from_numpy(img).float()
        # feats = self.feat(_path)





        # fn_white = self.white_fnames[index]
        # fn_white = fn_white if isinstance(self.white_fnames, list) else fn_white.as_py()
        # fn_white = self.text_compressor.decompress(fn_white)
        # path_or_fp_white = os.path.join(self.root_dir+'/white_back_img', fn_white)
        # _path_white = parse_path(path_or_fp_white)
        # feats_white = self.feat(_path_white)

        # img_white = cv2.imread(_path_white,cv2.IMREAD_GRAYSCALE)#这里用灰度图
        # # img_white = self.preprocess(img_white)
        # feats_white = torch.from_numpy(img_white).float()
        # # feats_white = self.postprocess(feats_white)

        # fn_mask = self.mask_fnames[index]
        # fn_mask = fn_mask if isinstance(self.mask_fnames, list) else fn_mask.as_py()
        # fn_mask = self.text_compressor.decompress(fn_mask)
        # path_or_fp_mask = os.path.join(self.root_dir+'/de-noise_img_crop', fn_mask)
        # _path_mask = parse_path(path_or_fp_mask)
        # feats_mask = self.feat(_path_mask)

        # img_mask = cv2.imread(_path_mask,cv2.IMREAD_GRAYSCALE)#这里用灰度图
        # # img_mask = self.preprocess(img_mask)
        # feats_mask = torch.from_numpy(img_mask).float()
        # feats_mask = self.postprocess(feats_mask)

        # return {"id": index, "img_source": feats,"img_white_source":feats_white,'img_mask_source':feats_mask}
        return {"id": index, "img_source": feats}

    def __len__(self):
        return len(self.fnames)

    # def size(self, index):
    #     return self.sizes[index]

    # def num_tokens(self, index):
    #     return self.size(index)
    
    def collater(self, samples):
        samples = [s for s in samples if s["img_source"] is not None]
        if len(samples) == 0:
            return {}
        sources = torch.stack([s["img_source"] for s in samples]) # (B,C,H,W)

        # sources_white = torch.stack([s["img_white_source"] for s in samples]) # (B,C,H,W)
        # img_mask_source = torch.stack([s["img_mask_source"] for s in samples]) # (B,C,H,W)
        input = {"img_source": sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        
        out["net_input"] = input
        return out
    def preprocess(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    def postprocess(self, feats):
        # todo image postprocess
        return feats


class ImageToTextDataset(FairseqDataset):
    '''
    与ImageToTextDataset的区别是这个不包含原文本，针对下游任务图像翻译
    '''
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        source_process = None,
        add_to_input=True,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.source = source_texts
        self.batch_targets = batch_targets
        self.add_to_input = add_to_input
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.source_process = source_process
        self.text_compressor = TextCompressor(level=text_compression_level)

        self.tgt_sizes = np.array([len(self.get_label(i, process_fn=self.process_label)) for i in range(len(self.dataset))])

    
    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)
    
    def get_source(self, index, process_fn=None):
        lbl = self.source[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = self.dataset[index]
        item['id'] = id
        item["label"] = self.get_label(index, process_fn=self.process_label)
        return item
    
    def __len__(self):
        return len(self.tgt_sizes)


    def size(self, index):
        src_size = self.src_sizes[index]
        tgt_size = self.tgt_sizes[index]
        return src_size, tgt_size

    def num_tokens(self, index):
        return self.tgt_sizes[index]

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
            if collated["net_input"].get("prev_output_tokens", None):
                collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                    collated["net_input"]["prev_output_tokens"],
                    pad_idx=self.pad,
                    left_pad=False,
                )
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        
        assert len(max_sizes) == 2
        ignored = indices[self.tgt_sizes[indices] > max_sizes[1]].tolist()
        indices = indices[self.tgt_sizes[indices] <= max_sizes[1]]

        return indices, ignored

