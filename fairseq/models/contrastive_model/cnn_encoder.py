# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from asyncio.log import logger
logger = logging.getLogger(__name__)
import math
from typing import Dict, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.contrastive_model.embeding import PositionEmbeddingSine, BoxRelationalEmbedding
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules import PositionalEmbedding

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class CnnEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, layers):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.max_source_positions = cfg.max_source_positions
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(1, stride=1)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.linear = nn.Linear(2048,256)
        self.encoder_layerdrop = cfg.encoder.layerdrop

        # self.grid_embedding = PositionEmbeddingSine(cfg.decoder.embed_dim // 2, normalize=True)
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )

        # self.embed_positions = (
        #     PositionalEmbedding(
        #         cfg.max_source_positions,
        #         2048,
        #         0,
        #         learned=cfg.encoder.learned_pos,
        #     )
        # )
        self.embed_positions = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print(self.training)
        # self.dnn = DnCNN()

        # self.craft = CRAFT()#检测网络
        # if self.training:#如果是训练，就加载预训练模型
        #     self.craft.load_state_dict(copyStateDict(torch.load(cfg.pre_model_path, map_location='cuda')))
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def get_pos_embedding(self, grids):
        bs = grids.shape[0]
        grid_embed = self.grid_embedding(grids)#这里不需要reshape，因为我们的cnn已经处理好格式了
        return grid_embed

    def forward(
        self,
        img_source,
        src_token = None,
        src_lengths = None,
    ):
        return self.forward_scriptable(img_source)


    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(self, img_source):
        # print('https://github.com/Yang-Liu1082/FDN.git',self.inplanes)
        img_source = img_source.permute(0,3,1,2)
        # res_img = img_source - img_mask_source#已有图片减去背景图片
        # x = self.dnn(res_img)
        # res_x = x

        # polys_list = get_boxes(self.craft,img_source)
        # batch_img_tensor_list,batch_img_white_tensor_list = batch_process_box(polys_list,img_source,img_white_source)
        # batch_img_tensor_list = shared_dnn(batch_img_tensor_list,self.dnn)



        #Resnet101
        x = self.conv1(img_source) # B x C x H x W
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x) # (B, C, H, W)


        #变形
        bz, channel, height, width = x.shape
        time_step = height * width
        x = x.view(bz, channel, time_step).permute(2,0,1).contiguous() # T x B x C
        x = self.linear(x)
        src_lengths = torch.ones(bz, device=x.device) * time_step
        padding_mask = torch.zeros((bz, time_step), device=x.device)

        #transformer
        for layer in self.layers:
            lr = layer(
                x,None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None
        
        # dummy_tokens = torch.ones((bz, time_step), device=img_source.device).int()
        # pos_embed = self.embed_positions(dummy_tokens).transpose(0,1).contiguous() # T x B x C
        # x = x #+ pos_embed
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "src_lengths": [src_lengths], # B
        }

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=False
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]


        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_padding_mask = []
        else:
            new_padding_mask = [(encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_padding_mask,
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        pass


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
