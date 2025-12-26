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

        encoder_layer = EncoderLayer_RGBD(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder_RGBD(encoder_layer, num_encoder_layers, encoder_norm)

        ## changes here
        encoder_layer_depth = EncoderLayer_RGBD(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm_depth = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_depth = TransformerEncoder_RGBD(encoder_layer_depth, num_encoder_layers, encoder_norm_depth)

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

        src2, src2_depth = self.self_attn(q, k, value=src, attn_mask=src_mask,
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
    
class RGBD_MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(
            self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias
        )
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.alpha = config["alpha"]
        self.beta = config["beta"]

        self.learnable_alpha_beta = bool(config.get("learnable_alpha_beta", False))
        if self.learnable_alpha_beta:
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha_raw = None
            self.beta_raw = None
    
    def get_alpha_beta(self):
        """Sigmoid関数で0-1の範囲に制約"""
        alpha_raw = torch.sigmoid(self.alpha_raw)
        beta_raw = torch.sigmoid(self.beta_raw)
        return alpha_raw, beta_raw

    def forward(self, img, dpt, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv_img = self.qkv_projection(img)
        qkv_dpt = self.qkv_projection(dpt)

        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query_img, key_img, value_img = torch.chunk(qkv_img, 3, dim=-1)
        query_dpt, key_dpt, value_dpt = torch.chunk(qkv_dpt, 3, dim=-1)

        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query_img.size()
        query_img = query_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key_img = key_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value_img = value_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

        # Calculate the attention scores
        attention_scores_img = torch.matmul(query_img, key_img.transpose(-1, -2))
        attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)

        batch_size, sequence_length, _ = query_dpt.size()
        query_dpt = query_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key_dpt = key_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value_dpt = value_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

        # Calculate the attention scores
        attention_scores_dpt = torch.matmul(query_dpt, key_dpt.transpose(-1, -2))
        attention_scores_dpt = attention_scores_dpt / math.sqrt(self.attention_head_size)

        # Softmax
        attention_probs_img = nn.functional.softmax(attention_scores_img, dim=-1)
        attention_probs_img = self.attn_dropout(attention_probs_img)

        attention_probs_dpt = nn.functional.softmax(attention_scores_dpt, dim=-1)
        attention_probs_dpt = self.attn_dropout(attention_probs_dpt)

        if not self.learnable_alpha_beta:
            ## Share-Fusion (probs is how atteentioned to each other)
            shared_attention_probs_img = (1-self.alpha)*attention_probs_img + self.alpha*attention_probs_dpt
            shared_attention_probs_dpt = (1-self.beta)*attention_probs_dpt + self.beta*attention_probs_img

        else:
            ## Share-Fusion ++
            
            alpha_raw, beta_raw = self.get_alpha_beta()
            shared_attention_probs_img = (1-alpha_raw)*attention_probs_img + alpha_raw*attention_probs_dpt
            shared_attention_probs_dpt = (1-beta_raw)*attention_probs_dpt + beta_raw*attention_probs_img

        # print(f"shared_attention_probs_img :{torch.sum(shared_attention_probs_img, dim=-1)}")
        # print(f"shared_attention_probs_dpt :{torch.sum(shared_attention_probs_dpt, dim=-1)}")

        # Calculate the attention output
        attention_output_img = torch.matmul(shared_attention_probs_img, value_img)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output_img = (
            attention_output_img.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )

        # Calculate the attention output
        attention_output_dpt = torch.matmul(shared_attention_probs_dpt, value_dpt)
        attention_output_dpt = (
            attention_output_dpt.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )

        # print(f"attention_output:{attention_output.shape}")
        # Project the attention output back to the hidden size
        attention_output_img = self.output_projection(attention_output_img)
        attention_output_img = self.output_dropout(attention_output_img)

        # print(f"attention_output:{attention_output.shape}")
        # Project the attention output back to the hidden size
        attention_output_dpt = self.output_projection(attention_output_dpt)
        attention_output_dpt = self.output_dropout(attention_output_dpt)

        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output_img, None, attention_output_dpt, None)
        else:
            # print(f"attention_probs.shape:{attention_probs_img.shape}")
            return (attention_output_img, shared_attention_probs_img, attention_output_dpt, shared_attention_probs_dpt)
        
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
