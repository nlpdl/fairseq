
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




class TextOnlyDataset(FairseqDataset):

    def __init__(
        self,
        labels,
        source_texts,
        pad,
        eos,
        batch_targets,
        process_label=None,
        source_process = None,
        add_to_input=True,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.labels = labels
        self.source = source_texts
        self.batch_targets = batch_targets
        self.add_to_input = add_to_input
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.source_process = source_process
        self.text_compressor = TextCompressor(level=text_compression_level)

        self.src_sizes = np.array([len(self.get_source(i, process_fn=self.process_label)) for i in range(len(self.source))])
        self.tgt_sizes = np.array([len(self.get_label(i, process_fn=self.process_label)) for i in range(len(self.labels))])

    
    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)
    
    def get_source(self, index, process_fn=None):
        lbl = self.source[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = {}
        item['id'] = index
        item["label"] = self.get_label(index, process_fn=self.process_label)
        item["source"] = self.get_source(index, process_fn=self.source_process)
        return item
    
    def __len__(self):
        return len(self.tgt_sizes)


    def size(self, index):
        src_size = self.src_sizes[index]
        # tgt_size = self.tgt_sizes[index]
        return src_size

    def num_tokens(self, index):
        return self.tgt_sizes[index]

    def collater(self, samples):
        collated = {"id": torch.LongTensor([s["id"] for s in samples])}
        # collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        source = [s["source"] for s in samples if s["id"] in indices]

        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            src_lengths = torch.LongTensor(
                [s["source"].ne(self.pad).long().sum() for s in samples]
            )
            src_token = [torch.cat([t, eos], axis=-1) for t in source]
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"] = {}
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens
            collated["net_input"]["src_token"] = data_utils.collate_tokens(
                                                src_token,
                                                pad_idx=self.pad,
                                                left_pad=True,
                                                eos_idx=eos,
                                                pad_to_length=None
                                            )

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
        collated["net_input"]["src_lengths"] = src_lengths
        return collated

    # def filter_indices_by_size(self, indices, max_sizes):
        
    #     assert len(max_sizes) == 2
    #     ignored = indices[self.tgt_sizes[indices] > max_sizes[1]].tolist()
    #     indices = indices[self.tgt_sizes[indices] <= max_sizes[1]]

    #     return indices, ignored

