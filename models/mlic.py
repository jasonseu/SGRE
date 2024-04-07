# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-9-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import copy
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

from .utils import Element_Wise_Layer
from .factory import register_model, create_backbone
from lib.util import get_loss_fn


__all__ = ['mlic']


class LowRankBilinearAttention(nn.Module):
    """
    Low-rank bilinear attention network.
    """
    def __init__(self, dim1, dim2, att_dim=2048):
        """
        :param dim1: feature size of encoded images
        :param dim2: feature size of encoded labels
        :param att_dim: size of the attention network
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)  # linear layer to transform encoded image
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)  # linear layer to transform decoder's output
        self.hidden_linear = nn.Linear(att_dim, att_dim)   # linear layer to calculate values to be softmax-ed
        self.target_linear = nn.Linear(att_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)  # softmax layer to calculate weights

    def forward(self, x1, x2, tau=1.0):
        """
        Forward propagation.
        :param 
            x1: a tensor of dimension (B, num_pixels, dim1)
            x2: a tensor of dimension (B, num_labels, dim2)
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  # (B, 1, num_pixels, att_dim)
        _x2 = self.linear2(x2).unsqueeze(dim=2)  # (B, num_labels, 1, att_dim)
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        temp = self.target_linear(t).squeeze(-1) # B, num_labels, num_pixels
        alpha = self.softmax(temp / tau) # (B, num_labels, num_pixels)
        return alpha
    
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))


def build_position_encoding(hidden_dim, arch, position_embedding, img_size):
    N_steps = hidden_dim // 2

    if arch in ['CvT_w24'] or 'vit' in arch:
        downsample_ratio = 16
    else:
        downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio, maxW=img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

    
class TransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, normalize_before, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderLayer(d_model, nhead, normalize_before=normalize_before)) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

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


class MLIC(nn.Module):
    def __init__(self, backbone, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.position_embedding = build_position_encoding(feat_dim, cfg.arch, 'sine', cfg.img_size)
        self.encoder = TransformerEncoder(feat_dim, cfg.num_heads, cfg.num_layers, cfg.normalize_before)
        if self.cfg.embed_type == 'random':
            self.embeddings = nn.Parameter(torch.empty((cfg.num_classes, 768)))
            nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
        else:
            self.embeddings = torch.from_numpy(np.load(cfg.embed_path)).float().cuda()
        text_dim = self.embeddings.shape[-1]
        self.attention = LowRankBilinearAttention(feat_dim, text_dim, 1024)
        self.fc = Element_Wise_Layer(cfg.num_classes, feat_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(2*feat_dim, 2*feat_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*feat_dim, cfg.num_classes)
        )
        self.criterion = get_loss_fn(cfg)
        self.net1 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.net3 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.net4 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y=None):
        x = self.backbone(x)
        if self.cfg.rm_cls and 'vit' in self.cfg.arch:
            x = x[:, 1:]
        pos = None
        if self.cfg.pos: 
            pos = self.position_embedding(x)
            pos = torch.flatten(pos, 2).transpose(1, 2)
        x = self.encoder(x, pos=pos)
        embeddings = self.embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)

        alpha = self.attention(x, embeddings, self.cfg.tau)
        f = torch.bmm(alpha, x)
        logits1 = self.fc(f)

        q = self.net1(x.transpose(1, 2)).transpose(1, 2)
        k = self.net2(f.transpose(1, 2))
        _alpha = torch.bmm(q, k)
        _alpha = F.softmax(_alpha, dim=-1)

        f = self.net3(f.transpose(1, 2)).transpose(1, 2)
        _x = torch.bmm(_alpha, f)
        _x = self.net4(_x.transpose(1, 2)).transpose(1, 2)

        x = torch.cat([x, _x], dim=-1)
        logits2 = self.fc2(x)
        logits2 = logits2.max(dim=1)[0]

        ce_loss = 0
        if self.training:
            bce1 = self.criterion(logits1, y)
            bce2 = self.criterion(logits2, y)
            ce_loss = bce2 + self.cfg.lamda * bce1
        
        return {
            'logits': logits2,
            'alpha': alpha,
            'ce_loss': ce_loss
        }
        

@register_model
def mlic(cfg):
    backbone, feat_dim = create_backbone(cfg.arch, img_size=cfg.img_size)
    model = MLIC(backbone, feat_dim, cfg)
    return model