# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging
from collections  import OrderedDict
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.models.transformer import TransformerEncoderBase
from fairseq.models.image_to_net_new import TransformerEncoder
from fairseq.models.image_to_net_new import TextEncoderPrenet
from fairseq.models.image_to_net_new import ImageFeatureExtraction
from fairseq.models.image_to_net_new import PretrainConfig
from .vgg_model import Model,CnnEncoderBase
from fairseq.modules import (
    GumbelVectorQuantizer,
)
logger = logging.getLogger(__name__)


class NewImageToNetPretrainModelBase(FairseqEncoderDecoderModel):
    def __init__(self, cfg, encoder, decoder,img_encoder_prenet,text_encoder_prenet,contrastive_encoder,contrastive_decoder):
        super().__init__(encoder, decoder)
        self.img_encoder_prenet = img_encoder_prenet
        if cfg.ocr_pretrain:
            logger.info('using pretrain ocr model')
            model_path = '/home/sxy/Projects/cp/base_data/model_v3/checkpoint/zh_sim_g2.pth'
            state_dict = torch.load(model_path,map_location='cpu')
            new_state_dict = OrderedDict()
            for key ,value in state_dict.items():
                if "Prediction" in key:
                    continue
                new_key = key[7:]
                new_state_dict[new_key] = value
            self.img_encoder_prenet.load_state_dict(new_state_dict)
        else:
            for m in self.img_encoder_prenet.modules():
                if isinstance(m,nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight,gain = 1)
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.remove_contrastive = cfg.remove_contrastive

        self.text_encoder_prenet = text_encoder_prenet
        self.contrastive_encoder = contrastive_encoder
        self.contrastive_decoder = contrastive_decoder
        self.if_meanpool = cfg.if_meanpool

        self.cfg = cfg
        self.supports_align_args = True
        self.taskname = cfg.taskname

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, PretrainConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --
        # cfg.encoder.cnnlayers = eval(cfg.encoder.cnnlayers)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens = cls.build_embedding(
            cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        )

        decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        img_encoder_prenet = cls.build_img_encoder_prenet(cfg)
        text_encoder_prenet = cls.build_text_encoder_prenet(encoder_embed_tokens, cfg)
        decoder = cls.build_decoder(cfg,tgt_dict, decoder_embed_tokens)
        encoder = cls.build_my_encoder(cfg)
        
        
        contrastive_encoder = cls.build_contrastive_encoder(cfg, src_dict, encoder_embed_tokens)
        contrastive_decoder = cls.build_decoder(cfg,tgt_dict, decoder_embed_tokens)

        return cls(cfg, encoder, decoder, img_encoder_prenet, text_encoder_prenet, contrastive_encoder,contrastive_decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb
    # Encoder Prenet
    @classmethod
    def build_text_encoder_prenet(cls, embed_tokens,cfg):
        return TextEncoderPrenet(embed_tokens, cfg)
    
    @classmethod
    def build_img_encoder_prenet(cls,cfg):
        if cfg.img_pre == 'resnet50':
            return CnnEncoderBase(cfg.encoder_embed_dim,layers = [3, 4, 6, 3])
        elif cfg.img_pre == 'resnet101':
            return CnnEncoderBase(cfg.encoder_embed_dim,layers = [3, 4, 23, 3])
        else:
            return Model(cfg.input_channel,cfg.output_channel,cfg.output_channel)


    @classmethod
    def build_my_encoder(cls, cfg,embed_tokens=None):
        return TransformerEncoder(cfg,embed_tokens)
    @classmethod
    def build_contrastive_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    # def load_state_dict(
    #     self,
    #     state_dict,
    #     strict=True,
    #     model_cfg=None,
    #     args=None,
    # ):
    #     """
    #     预加载对比学习。
    #     """
        
    #     model_state_dict = self.state_dict()
        
    #     initialized_keys = []
    #     for key in state_dict:
    #         # if 'decoder.' in key:#不要decoder的参数
    #         #     continue
    #         if 'encoder.' in key:
    #             temp_key = key.replace('encoder.','contrastive_encoder.')
    #         if 'decoder.' in key:
    #             temp_key = key.replace('decoder.','contrastive_decoder.')
    #         #更改指向，这里contrastive_encoder才是原版encoder，需要对比学习的地方
    #         #从头开始训练的时候才用
    #         if temp_key in model_state_dict:  
    #             # 对应参数位置进行替换
    #             model_state_dict[temp_key] = state_dict[key].to(model_state_dict[temp_key].device)
    #             initialized_keys.append(key)
    #     logger.info(f"Keys initialized with pretrained model: {initialized_keys}")
    #     logger.info(f"coverage percent: {len(initialized_keys) / len(model_state_dict)}")  
            
    #     return super().load_state_dict(model_state_dict, strict, model_cfg, args)
    


    # def load_state_dict(
    #     self,
    #     state_dict,
    #     strict=True,
    #     model_cfg=None,
    #     args=None,
    # ):
    #     """
        
    #     """
        
    #     model_state_dict = self.state_dict()
        
    #     initialized_keys = []
    #     for key in state_dict:
    #         # if 'decoder.' in key:#不要decoder的参数
    #         #     continue
    #         if 'encoder' in key:
    #             continue
    #         #更改指向，这里contrastive_encoder才是原版encoder，需要对比学习的地方
    #         #从头开始训练的时候才用
    #         if temp_key in model_state_dict:  
    #             # 对应参数位置进行替换
    #             model_state_dict[key] = state_dict[key].to(model_state_dict[key].device)
    #             initialized_keys.append(key)
    #     logger.info(f"Keys initialized with pretrained model: {initialized_keys}")
    #     logger.info(f"coverage percent: {len(initialized_keys) / len(model_state_dict)}")  
            
    #     return super().load_state_dict(model_state_dict, strict, model_cfg, args)

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        """
        Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        
        model_state_dict = self.state_dict()
        
        initialized_keys = []
        for key in state_dict:
            if key in model_state_dict:  
                model_state_dict[key] = state_dict[key].to(model_state_dict[key].device)
                initialized_keys.append(key)
        logger.info(f"Keys initialized with pretrained model: {initialized_keys}")
        logger.info(f"coverage percent: {len(initialized_keys) / len(model_state_dict)}")  
            
        return super().load_state_dict(model_state_dict, strict, model_cfg, args)


    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        img_source=None,
        src_token=None,
        src_lengths=None,
        prev_output_tokens=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        #先判定输入性质

        encoder_input, encoder_padding_mask = self.img_encoder_prenet(img_source)

        #图像有对比学习部分
        if self.if_meanpool:
            if self.remove_contrastive:
                encoder_out = self.encoder(encoder_input, encoder_padding_mask)
                e_out = torch.mean(encoder_out['encoder_out'][0].transpose(0,1),dim=1)

            else:
                encoder_out = self.encoder(encoder_input, encoder_padding_mask)
                # print('img',encoder_out)
                contrastive_encoder_out = self.contrastive_encoder(src_token, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
                c_out = torch.mean(contrastive_encoder_out['encoder_out'][0].transpose(0,1),dim=1)
                e_out = torch.mean(encoder_out['encoder_out'][0].transpose(0,1),dim=1)
        else:
            encoder_out = self.encoder(encoder_input, encoder_padding_mask)
            contrastive_encoder_out = self.contrastive_encoder(src_token, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
            c_out = contrastive_encoder_out['encoder_out'][0]
            e_out = encoder_out['encoder_out'][0]

            e_out = e_out.transpose(0,2)
            e_out = nn.Linear(e_out.size(-1),c_out.size(0)).cuda().half()(e_out)
            e_out = e_out.transpose(0,2).transpose(0,1)
            c_out = c_out.transpose(0,1)

        
        
        
       
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        if self.remove_contrastive is False:
            contrastive_decoder_out = self.contrastive_decoder(
                prev_output_tokens,
                encoder_out=contrastive_encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )

            decoder_out[1]['contrastive_encoder_out'] = c_out
            decoder_out[1]['encoder_out'] = e_out
            decoder_out[1]['contrastive_decoder_out'] = contrastive_decoder_out[0]
        return decoder_out

    def forward_mt(
        self,
        img_source=None,
        src_token=None,
        src_lengths=None,
        prev_output_tokens=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        encoder_input, encoder_padding_mask = self.text_encoder_prenet(src_token)
        encoder_out = self.encoder(encoder_input, encoder_padding_mask)
        contrastive_encoder_out = self.contrastive_encoder(src_token, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        e_out = encoder_out['encoder_out'][0].transpose(0,1)
        c_out = contrastive_encoder_out['encoder_out'][0].transpose(0,1)

        
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        # print(encoder_out['encoder_out'][0].transpose(0,1).shape)
        decoder_out[1]['encoder_out'] = e_out
        decoder_out[1]['contrastive_encoder_out'] = c_out

        return decoder_out


        
        # print(decoder_out[0].shape)
        # assert 1 == 0
        

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m