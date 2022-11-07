from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import contextlib
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    TransformerPretrainEncoderLayer,
)
from torch import Tensor
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model) 
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v


    def forward(self, pos_seq):
        pos_seq[pos_seq < -self.maxlen] = -self.maxlen
        pos_seq[pos_seq >= self.maxlen] = self.maxlen - 1
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None

class TransformerEncoder(FairseqEncoder):
    def __init__(self, cfg, embed_tokens=None):
        self.cfg = cfg
        super().__init__(None)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = cfg.encoder_layerdrop
        self.freeze_encoder_updates = cfg.freeze_encoder_updates
        if cfg.no_freeze_encoder_layer is not None:
            self.no_freeze_encoder_layer = eval(cfg.no_freeze_encoder_layer)
        else:
            self.no_freeze_encoder_layer = None
        self.num_updates = 0
        export = getattr(cfg, "export", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.use_sent_enc_layer = cfg.use_sent_enc_layer
        self.unb_enc_layer = getattr(cfg, "unb_enc_layer", -1)

        self.layer_norm_first = cfg.layer_norm_first
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim, eps=cfg.layer_norm_eps, export=export)
        
        if cfg.relative_position_embedding:
            self.pos_emb = RelativePositionalEncoding(cfg.encoder_embed_dim//cfg.encoder_attention_heads, cfg.encoder_max_relative_position)


    def build_encoder_layer(self, cfg):
        layer = TransformerPretrainEncoderLayer(cfg)
        return layer

    def forward(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        #不知道有没有用，先写上，冻结参数
        if self.no_freeze_encoder_layer is None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.forward_scriptable(
                encoder_in, encoder_padding_mask, return_all_hiddens, tgt_layer=tgt_layer,
            )

        return encoder_out

    def forward_scriptable(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        if self.no_freeze_encoder_layer is not None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # compute padding mask
            if not self.use_sent_enc_layer:
                has_pads = encoder_in.device.type == "xla" or encoder_padding_mask.any()

            if not self.layer_norm_first:
                encoder_in = self.layer_norm(encoder_in)

            encoder_in = self.dropout_module(encoder_in)

            # B x T x C -> T x B x C
            x = encoder_in.transpose(0, 1)

            encoder_states = []

            if return_all_hiddens:
                encoder_states.append(x)

            ## relative position embedding
            if self.cfg.relative_position_embedding:
                x_len = x.shape[0]
                pos_seq = torch.arange(0, x_len).long().to(x.device)
                pos_seq = pos_seq[:, None] - pos_seq[None, :]
                pos_k, pos_v = self.pos_emb(pos_seq)
            else:
                pos_k = None

        # encoder layers
        r = None
        d = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()

            with torch.no_grad() if (not ft) and i not in self.no_freeze_encoder_layer else contextlib.ExitStack():
                if not self.training or (dropout_probability > self.encoder_layerdrop) or i == self.unb_enc_layer:
                    if self.use_sent_enc_layer:
                        x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, self_attn_mask=None, need_weights=False, pos_bias=pos_k)
                        # x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, need_weights=False, pos_bias=pos_k)
                    else:
                        x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=None)
                        # x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)
                if i == self.unb_enc_layer:
                    d = x

                if i == tgt_layer:
                    r = x
                    break

                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        with torch.no_grad() if not ft else contextlib.ExitStack():
            # Finally T x B x C
            if self.layer_norm_first:
                x = self.layer_norm(x.transpose(0, 1)).transpose(0, 1)

            if r is not None:
                x = r
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "decoder_input": [d],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["decoder_input"]) == 0 or encoder_out["decoder_input"][0] is None:
            new_decoder_input = []
        else:
            new_decoder_input = [
                encoder_out["decoder_input"][0].index_select(0, new_order)
            ]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "decoder_input": new_decoder_input,
        }

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     for i in range(self.num_layers):
    #         # update layer norms
    #         if not isinstance(self.layers[i], TransformerSentenceEncoderLayer):
    #             self.layers[i].upgrade_state_dict_named(
    #                 state_dict, "{}.layers.{}".format(name, i)
    #             )

    #     version_key = "{}.version".format(name)
    #     if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
    #         # earlier checkpoints did not normalize after the stack of layers
    #         self.layer_norm = None
    #         self.normalize = False
    #         state_dict[version_key] = torch.Tensor([1])
    #     return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates