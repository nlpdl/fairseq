# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.contrastive_model import ContrastiveConfig, CnnEncoderBase
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.models.transformer import TransformerEncoderBase


logger = logging.getLogger(__name__)


class ContrastiveModelBase(FairseqEncoderDecoderModel):
    """

    Args:
        encoder (CnnEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder,contrastive_encoder):
        super().__init__(encoder, decoder)
        self.contrastive_encoder = contrastive_encoder

        # if self.training:#如果是训练，就加载预训练模型
            # self.craft.load_state_dict(copyStateDict(torch.load(cfg.pre_model_path, map_location='cuda')))
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, ContrastiveConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --
        cfg.encoder.cnnlayers = eval(cfg.encoder.cnnlayers)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens = cls.build_embedding(
            cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        )

        decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        
        # if cfg.share_all_embeddings:
        #     if src_dict != tgt_dict:
        #         raise ValueError("--share-all-embeddings requires a joined dictionary")
        #     if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
        #         raise ValueError(
        #             "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
        #         )
        #     if cfg.decoder.embed_path and (
        #         cfg.decoder.embed_path != cfg.encoder.embed_path
        #     ):
        #         raise ValueError(
        #             "--share-all-embeddings not compatible with --decoder-embed-path"
        #         )
        #     encoder_embed_tokens = cls.build_embedding(
        #         cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        #     )
        #     decoder_embed_tokens = encoder_embed_tokens
        #     cfg.share_decoder_input_output_embed = True
        # elif cfg.merge_src_tgt_embed:
        #     logger.info(f"source dict size: {len(src_dict)}")
        #     logger.info(f"target dict size: {len(tgt_dict)}")
        #     src_dict.update(tgt_dict)
        #     task.src_dict = src_dict
        #     task.tgt_dict = src_dict
        #     logger.info(f"merged dict size: {len(src_dict)}")
        #     encoder_embed_tokens = cls.build_embedding(
        #         cfg, src_dict, cfg.encoder.embed_dim
        #     )
        #     decoder_embed_tokens = encoder_embed_tokens
        #     cfg.share_decoder_input_output_embed = True
        # else:
        #     encoder_embed_tokens = cls.build_embedding(
        #         cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
        #     )
        #     decoder_embed_tokens = cls.build_embedding(
        #         cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
        #     )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        
        encoder = cls.build_encoder(cfg, src_dict, cfg.encoder.cnnlayers)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        contrastive_encoder = cls.build_contrastive_encoder(cfg, src_dict, encoder_embed_tokens)
        return cls(cfg, encoder, decoder,contrastive_encoder)

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

    @classmethod
    def build_encoder(cls, cfg, src_dict, layers):
        return CnnEncoderBase(cfg, src_dict, layers)
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
    #     Copies parameters and buffers from *state_dict* into this module and
    #     its descendants.

    #     Overrides the method in :class:`nn.Module`. Compared with that method
    #     this additionally "upgrades" *state_dicts* from old checkpoints.
    #     """
        
    #     model_state_dict = self.state_dict()
        
    #     initialized_keys = []
    #     for key in state_dict:
    #         temp_key = key.replace('encoder.','contrastive_encoder.')
    #         #更改指向，这里contrastive_encoder才是原版encoder，需要对比学习的地方
    #         #从头开始训练的时候才用
    #         if temp_key in model_state_dict:  
    #             # 对应参数位置进行替换
    #             model_state_dict[temp_key] = state_dict[key].to(model_state_dict[temp_key].device)
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
        img_source,
        src_token,
        src_lengths,
        prev_output_tokens,
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
        # print(src_token.shape)
        contrastive_encoder_out = self.contrastive_encoder(src_token, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        encoder_out = self.encoder(img_source)
        src_lengths = encoder_out["src_lengths"]
        # batch_img_tensor_list = encoder_out["batch_img_tensor_list"]
        # batch_img_white_tensor_list = encoder_out["batch_img_white_tensor_list"]
        # img_loss = calculation_loss(batch_img_tensor_list,batch_img_white_tensor_list,self.loss)

        c_out = contrastive_encoder_out['encoder_out'][0]

        e_out = encoder_out['encoder_out'][0].transpose(0,2)

        e_out = nn.Linear(e_out.size(-1),c_out.size(0)).cuda()(e_out)

        e_out = e_out.transpose(0,2)


        

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out[1]['contrastive_encoder_out'] = c_out
        decoder_out[1]['encoder_out'] = e_out
        # print(decoder_out[0].shape)
        # assert 1 == 0
        return decoder_out

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

# def calculation_loss(batch_img_tensor_list,batch_img_white_tensor_list,loss):
#     sum_loss = 0
#     count = [len(x) for x in batch_img_tensor_list]
#     total_num = sum(count)#计算总元素个数
#     for batch_img_tensor,batch_img_white_tensor in zip(batch_img_tensor_list,batch_img_white_tensor_list):
#         for img_tensor,img_white_tensor in zip(batch_img_tensor,batch_img_white_tensor):
#             sum_loss = sum_loss + loss(img_tensor,img_white_tensor)
#     return sum_loss/total_num