
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding


class TextEncoderPrenet(nn.Module):
    def __init__(
        self,
        embed_tokens,
        args,
    ):
        super(TextEncoderPrenet, self).__init__()
        self.padding_idx = embed_tokens.padding_idx
        pos_enc_class = (
            ScaledPositionalEncoding
        )

        self.encoder_prenet = nn.Sequential(
            embed_tokens,
            pos_enc_class(args.encoder_embed_dim, args.transformer_enc_positional_dropout_rate, max_len=args.max_source_positions),
        )

    def forward(self, src_tokens):
        return self.encoder_prenet(src_tokens), src_tokens.eq(self.padding_idx)