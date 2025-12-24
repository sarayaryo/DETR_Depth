import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

class Transformer_RGBD(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer_RGBD(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ## changes here
        encoder_layer_depth = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm_depth = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_depth = TransformerEncoder(encoder_layer_depth, num_encoder_layers, encoder_norm_depth)

        ## changes here
        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(d_model * 2, d_model * 2),  # 512 -> 512
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 2, d_model)  # 512 -> 256
        # )


        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, src_depth=None, pos_embed_depth=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        ## changes here
        if src_depth is not None:
            src_depth = src_depth.flatten(2).permute(2, 0, 1)
            if pos_embed_depth is not None:
                pos_embed_depth = pos_embed_depth.flatten(2).permute(2, 0, 1)
            else:
                pos_embed_depth = pos_embed
            memory_depth = self.encoder_depth(src_depth, src_key_padding_mask=mask, pos=pos_embed_depth)
            # fusion  Concatenate: (HW, B, 256) + (HW, B, 256) -> (HW, B, 512)
            # memory_concat = torch.cat((memory, memory_depth), dim=2)  
            # MLP Fusion: (HW, B, 512) -> (HW, B, 256)
            # memory_fused = self.fusion_mlp(memory_concat)
            # memory = memory_fused
            memory = memory + memory_depth  # Simple addition fusion
            
        else:
            memory_depth = None


        hs, attn_weights = self.decoder(tgt, memory, memory_key_padding_mask=mask, ## changes here
                          pos=pos_embed, query_pos=query_embed)
        
        ## changes here
        if memory_depth is not None:
            return hs.transpose(1, 2), attn_weights, memory.permute(1, 2, 0).view(bs, c, h, w), memory_depth.permute(1, 2, 0).view(bs, c, h, w)
        else:
            return hs.transpose(1, 2), attn_weights, memory.permute(1, 2, 0).view(bs, c, h, w) ## changes here  
        
class TransformerEncoder_RGBD(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_depth,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        output_depth = src_depth

        for layer in self.layers:
            output = layer(output, output_depth, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class EncoderLayer_RGBD(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_depth,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        q_depth = k_depth = self.with_pos_embed(src_depth, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2_depth = self.self_attn(q_depth, k_depth, value=src_depth, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
