import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List, Tuple
from models.transformer import TransformerDecoder, TransformerDecoderLayer

class Transformer_RGBD(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()

        encoder_layer = EncoderLayer_RGBD(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, args=args)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder_RGBD(encoder_layer, num_encoder_layers, encoder_norm)

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

        # self._reset_parameters()

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

        if src_depth is not None:
            src_depth = src_depth.flatten(2).permute(2, 0, 1)
            if pos_embed_depth is not None:
                pos_embed_depth = pos_embed_depth.flatten(2).permute(2, 0, 1)
            else:
                pos_embed_depth = pos_embed

        tgt = torch.zeros_like(query_embed)
        memory, memory_depth = self.encoder(src, src_depth, src_key_padding_mask=mask, pos=pos_embed, pos_depth=pos_embed_depth)
        memory = memory + memory_depth 

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
        self.norm_depth = norm

    def forward(self, src, src_depth,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pos_depth: Optional[Tensor] = None):
        output = src
        output_depth = src_depth

        for layer in self.layers:
            output, output_depth = layer(output, output_depth, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, pos_depth=pos_depth)
        if self.norm is not None:
            output = self.norm(output)
            output_depth = self.norm_depth(output_depth)

        return output, output_depth

class EncoderLayer_RGBD(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.self_attn = RGBD_MultiHeadAttention(d_model, nhead, dropout=dropout, args=args)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        ## --Depth Stream-- 
        self.linear1_depth = nn.Linear(d_model, dim_feedforward)
        self.dropout_depth = nn.Dropout(dropout)
        self.linear2_depth = nn.Linear(dim_feedforward, d_model)
        
        self.norm1_depth = nn.LayerNorm(d_model)
        self.norm2_depth = nn.LayerNorm(d_model)
        self.dropout1_depth = nn.Dropout(dropout)
        self.dropout2_depth = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_depth,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     pos_depth: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        q_depth = k_depth = self.with_pos_embed(src_depth, pos_depth)

        src2, src_depth2, _, _ = self.self_attn(q, k, src, q_depth, k_depth, src_depth,attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ---- Depth Stream ----
        src_depth = src_depth + self.dropout1_depth(src_depth2)
        src_depth = self.norm1_depth(src_depth)
        src_depth2 = self.linear2_depth(self.dropout_depth(self.activation(self.linear1_depth(src_depth))))
        src_depth = src_depth + self.dropout2_depth(src_depth2)
        src_depth = self.norm2_depth(src_depth)

        return src, src_depth

    def forward_pre(self, src, src_depth,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, 
                    pos_depth: Optional[Tensor] = None):
        # RGB Stream
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)        
        # Depth Stream
        src_depth2 = self.norm1_depth(src_depth)
        q_depth = k_depth = self.with_pos_embed(src_depth2, pos_depth)        
        # RGBD Attention
        src2, src_depth2, _, _ = self.self_attn(
            q, k, src2, 
            q_depth, k_depth, src_depth2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        # Residual connections
        src = src + self.dropout1(src2)
        src_depth = src_depth + self.dropout1_depth(src_depth2)
        
        # Feedforward - RGB
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        # Feedforward - Depth
        src_depth2 = self.norm2_depth(src_depth)
        src_depth2 = self.linear2_depth(self.dropout_depth(self.activation(self.linear1_depth(src_depth2))))
        src_depth = src_depth + self.dropout2_depth(src_depth2)
        
        return src, src_depth

    def forward(self, src, src_depth,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pos_depth: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_depth, src_mask, src_key_padding_mask, pos, pos_depth)
        return self.forward_post(src, src_depth, src_mask, src_key_padding_mask, pos, pos_depth)


# class RGBD_MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, alpha=0.5, beta=0.5):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.alpha = alpha
#         self.beta = beta

#         self.rgb_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)
#         self.depth_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)


#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         # --- RGB Stream Layers ---
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#         # --- Depth Stream Layers ---
#         self.q_proj_depth = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.k_proj_depth = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj_depth = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj_depth = nn.Linear(embed_dim, embed_dim, bias=bias)

#         self.dropout = nn.Dropout(dropout)

#     from typing import Optional, Tuple
#     def forward(
#         self,
#         query: Tensor,           # RGB Query
#         key: Tensor,             # RGB Key
#         value: Tensor,           # RGB Value
#         query_depth: Tensor,     # Depth Query
#         key_depth: Tensor,       # Depth Key
#         value_depth: Tensor,     # Depth Value
#         key_padding_mask: Optional[Tensor] = None,
#         attn_mask: Optional[Tensor] = None,
#         need_weights: bool = True,
#     ) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        
#         if self.batch_first:
#             # (Batch, Len, Dim) ->そのままでOK
#             pass
#         else:
#             # (Len, Batch, Dim) -> (Batch, Len, Dim) に統一して計算
#             query = query.transpose(0, 1)
#             key = key.transpose(0, 1)
#             value = value.transpose(0, 1)
#             query_depth = query_depth.transpose(0, 1)
#             key_depth = key_depth.transpose(0, 1)
#             value_depth = value_depth.transpose(0, 1)

#         B, L, _ = query.shape
#         _, S, _ = key.shape

#         # 2. 射影 (Projection) & Head分割
#         # Shape: (Batch, Heads, Len, HeadDim)
        
#         # --- RGB ---
#         q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

#         # --- Depth ---
#         q_d = self.q_proj_depth(query_depth).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
#         k_d = self.k_proj_depth(key_depth).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
#         v_d = self.v_proj_depth(value_depth).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

#         # 3. Attention Score (Scaled Dot-Product)
#         # (Batch, Heads, L, S)
#         scale = math.sqrt(self.head_dim)
#         attn_weights_img = torch.matmul(q, k.transpose(-2, -1)) / scale
#         attn_weights_dpt = torch.matmul(q_d, k_d.transpose(-2, -1)) / scale

#         # 4. Mask 適用
#         # key_padding_mask: (Batch, S) -> Trueの部分を無視
#         if key_padding_mask is not None:
#             # (Batch, 1, 1, S) に拡張してマスク
#             mask = key_padding_mask.view(B, 1, 1, S)
#             attn_weights_img = attn_weights_img.masked_fill(mask, float('-inf'))
#             attn_weights_dpt = attn_weights_dpt.masked_fill(mask, float('-inf'))

#         # attn_mask: (L, S) など
#         if attn_mask is not None:
#             attn_weights_img += attn_mask
#             attn_weights_dpt += attn_mask

#         # 5. Softmax & Dropout
#         attn_probs_img = F.softmax(attn_weights_img, dim=-1)
#         attn_probs_img = self.dropout(attn_probs_img)

#         attn_probs_dpt = F.softmax(attn_weights_dpt, dim=-1)
#         attn_probs_dpt = self.dropout(attn_probs_dpt)

#         # 6. Share-Fusion
#         shared_probs_img = (1 - self.alpha) * attn_probs_img + self.alpha * attn_probs_dpt
#         shared_probs_dpt = (1 - self.beta) * attn_probs_dpt + self.beta * attn_probs_img

#         # 7. Valueとの積
#         # (Batch, Heads, L, S) x (Batch, Heads, S, HeadDim) -> (Batch, Heads, L, HeadDim)
#         output_img = torch.matmul(shared_probs_img, v)
#         output_dpt = torch.matmul(shared_probs_dpt, v_d)

#         # 8. 結合 & 出力射影
#         # (Batch, Heads, L, HeadDim) -> (Batch, L, Dim)
#         output_img = output_img.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
#         output_dpt = output_dpt.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

#         output_img = self.out_proj(output_img)
#         output_dpt = self.out_proj_depth(output_dpt)

#         # 戻り値: ((RGB出力, Depth出力), Attention重み(代表してRGBまたはNone))
#         # EncoderLayer側の期待する受け取り方に合わせて調整してください
#         return output_img, output_dpt, shared_probs_img, shared_probs_dpt
    

class RGBD_MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, args=None):
        super().__init__()
        self.rgb_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)
        self.depth_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        # Fusionパラメータ
        if args.use_learnable_param:
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = 0.5
            self.beta = 0.5
    
    def forward(
        self,
        query: Tensor, key: Tensor, value: Tensor,           # RGB Inputs
        query_depth: Tensor, key_depth: Tensor, value_depth: Tensor, # Depth Inputs
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        
        # --- 1. Q, K, V の射影 (Projection) ---
        # 標準の in_proj_weight を使って計算する (ここがポイント！)
        # F.linear(input, weight, bias) を使う
        
        # RGB Stream
        q_rgb, k_rgb, v_rgb = self._project_qkv(query, key, value, self.rgb_attn)
        
        # Depth Stream
        q_dpt, k_dpt, v_dpt = self._project_qkv(query_depth, key_depth, value_depth, self.depth_attn)

        # --- 2. Scaled Dot-Product Attention ---
        # (Batch, Heads, Len, HeadDim) の形式に変換済み
        scale = self.head_dim ** -0.5
        
        # Attention Scores: Q * K^T
        attn_weights_rgb = torch.matmul(q_rgb, k_rgb.transpose(-2, -1)) * scale
        attn_weights_dpt = torch.matmul(q_dpt, k_dpt.transpose(-2, -1)) * scale

        # Mask処理 (Key Padding Mask & Attn Mask)
        if key_padding_mask is not None:
            # key_padding_mask: (Batch, S) -> (Batch, 1, 1, S)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights_rgb = attn_weights_rgb.masked_fill(mask, float('-inf'))
            attn_weights_dpt = attn_weights_dpt.masked_fill(mask, float('-inf'))
        
        if attn_mask is not None:
            attn_weights_rgb += attn_mask
            attn_weights_dpt += attn_mask

        # Softmax & Dropout
        attn_probs_rgb = F.softmax(attn_weights_rgb, dim=-1)
        attn_probs_rgb = F.dropout(attn_probs_rgb, p=self.rgb_attn.dropout, training=self.training)
        
        attn_probs_dpt = F.softmax(attn_weights_dpt, dim=-1)
        attn_probs_dpt = F.dropout(attn_probs_dpt, p=self.depth_attn.dropout, training=self.training)

        # --- 3. Share-Fusion (確率分布の混合) ---
        shared_probs_rgb = (1 - self.alpha) * attn_probs_rgb + self.alpha * attn_probs_dpt
        shared_probs_dpt = (1 - self.beta) * attn_probs_dpt + self.beta * attn_probs_rgb

        # --- 4. Valueとの積 & 出力射影 ---
        output_rgb = torch.matmul(shared_probs_rgb, v_rgb)
        output_dpt = torch.matmul(shared_probs_dpt, v_dpt)

        # Reshape & Output Projection
        output_rgb = self._concat_projection(output_rgb, self.rgb_attn)
        output_dpt = self._concat_projection(output_dpt, self.depth_attn)

        return output_rgb, output_dpt, shared_probs_rgb, shared_probs_dpt

    def _project_qkv(self, q, k, v, attn_layer):
            """
            修正版: Q, K, V が異なる場合（DETR Encoderは V != Q）に対応
            """
            # (Len, Batch, Dim) -> (Batch, Len, Dim)
            tgt_len, bsz, embed_dim = q.shape
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

            # in_proj_weight (3*Dim, Dim) を持っている場合
            if hasattr(attn_layer, 'in_proj_weight') and attn_layer.in_proj_weight is not None:
                # 重みとバイアスを3等分する (Q用, K用, V用)
                w_q, w_k, w_v = attn_layer.in_proj_weight.chunk(3, dim=0)
                b_q, b_k, b_v = attn_layer.in_proj_bias.chunk(3, dim=0)
                
                # それぞれ個別に射影する
                # ※ DETRでは q=k=(src+pos), v=src なので、qとvが違う！
                q_proj = F.linear(q, w_q, b_q)
                k_proj = F.linear(k, w_k, b_k)
                v_proj = F.linear(v, w_v, b_v)
                
            else:
                # 互換性のため（q_proj_weightなどが分かれている場合）
                q_proj = F.linear(q, attn_layer.q_proj_weight, attn_layer.q_proj_bias)
                k_proj = F.linear(k, attn_layer.k_proj_weight, attn_layer.k_proj_bias)
                v_proj = F.linear(v, attn_layer.v_proj_weight, attn_layer.v_proj_bias)

            # Head分割 & Transpose
            # (Batch, Len, Dim) -> (Batch, Len, Heads, HeadDim) -> (Batch, Heads, Len, HeadDim)
            q_proj = q_proj.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_proj = k_proj.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_proj = v_proj.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            return q_proj, k_proj, v_proj

    def _concat_projection(self, output, attn_layer):
        # (Batch, Heads, Len, HeadDim) -> (Batch, Len, Heads, HeadDim) -> (Batch, Len, Dim)
        bsz, heads, tgt_len, head_dim = output.shape
        output = output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        # Out Projection
        output = F.linear(output, attn_layer.out_proj.weight, attn_layer.out_proj.bias)
        
        # (Batch, Len, Dim) -> (Len, Batch, Dim) に戻す
        output = output.transpose(0, 1)
        return output
       
def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_RGBD(args):
    return Transformer_RGBD(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args
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
