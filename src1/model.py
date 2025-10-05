import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DecoderLayerSimple(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x):
        "Follow Figure 1 (right) for connections."
        # We can not handle more than one input, so just generate a random one here
        # m = torch.sqrt(1 - x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SimpleGPT2(nn.Module):
    def __init__(self, N=6, d_model=512, h=8, dropout=0.1):
        """A simplified bert without embedding and language model heads"""
        super().__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        layer = DecoderLayerSimple(d_model, c(attn), c(attn), c(ff), dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.reduce = lambda x: x.sum(-1)

    def forward(self, x):
        """
        Args:
            x (Tensor[batch size, sequence length, d_model]): input after encoder

        Returns:
            out ()
        """
        print(x.shape)
        for layer in self.layers:
            x = layer(x)
        return self.reduce(self.norm(x))

    def join_layers(self):
        return [i for i in self.layers] + [self.norm, self.reduce]


def make_simplegpt2(*args) -> SimpleGPT2:
    return SimpleGPT2(*args)


def test():
    device = "cuda:0"

    d_model = 32
    n_layer = 4
    model = make_simplegpt2(n_layer, d_model).to(device)

    batch_size = 4
    seq_len = 12
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    out = model(x)
    print(f"Input shape: {x.shape}; \n" f"Output shape: {out.shape}")
