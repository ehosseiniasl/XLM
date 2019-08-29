# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# !/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# sys.append_path('utils')
from src.model.utils.adaptive_span import AdaptiveSpan
import ipdb
import ast
import numpy as np
from .memory import HashingMemory
from .transformer import PredLayer

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span


def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, hidden_size, attn_span,
                 dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span,
                                              **adapt_span_params, **kargs)

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class MultiHeadSeqAttentionPersistentMem(nn.Module):
    def __init__(self, hidden_size, nb_heads, inner_hidden_size, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        self.persistent_keys = nn.Parameter(torch.randn((1, hidden_size, inner_hidden_size)), requires_grad=True)
        self.persistent_values = nn.Parameter(torch.randn((1, inner_hidden_size, hidden_size)), requires_grad=True)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):

        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)
        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        persistent_keys = self.head_reshape(self.persistent_keys.transpose(-1,-2))
        persistent_values = self.head_reshape(self.persistent_values)
        out_persistent = []
        for i in range(0, query.shape[0], self.nb_heads):
            out_persistent.append(self.attn(query[i: i+self.nb_heads], persistent_keys, persistent_values, key_pe))
        out_persistent = torch.cat(out_persistent)
        out += out_persistent
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class TransformerSeqLayerPersistentMem(nn.Module):
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttentionPersistentMem(hidden_size=hidden_size, **kargs)
        # self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        # self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        # ff_out = self.ff(h)
        # out = self.norm2(h + ff_out)  # B x M x H
        out = h
        return out


class TransformerSeqLayerPKM(nn.Module):
    def __init__(self, hidden_size, mem_enc_type, params, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.mem_enc_type = mem_enc_type
        if 'in' not in mem_enc_type: # FFN used instead of in_memory
            self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        # build memory layer
        self.memories = nn.ModuleDict()
        for v in mem_enc_type:
            self.memories[f'{v}'] = HashingMemory.build(input_dim=hidden_size, output_dim=hidden_size, params=params)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        if 'in' in self.mem_enc_type:
            out = self.memories['in'](h)
        else:
            assert hasattr(self, 'ff')
            out = self.ff(h)
        # ff_out = self.ff(h)
        out = self.norm2(h + out)  # B x M x H

        if 'after' in self.memories.keys():
            out = self.memories['after'](out)
        return out


class TransformerSeqLayerPersistentMemPKM(nn.Module):
    def __init__(self, hidden_size, mem_enc_type, params, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttentionPersistentMem(hidden_size=hidden_size, **kargs)
        # self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.mem_enc_type = mem_enc_type
        # if 'in' not in mem_enc_type:  # FFN used instead of in_memory
        #     self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        # build memory layer
        self.memories = nn.ModuleDict()
        for v in mem_enc_type:
            self.memories[f'{v}'] = HashingMemory.build(input_dim=hidden_size, output_dim=hidden_size, params=params)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        # ff_out = self.ff(h)
        # out = self.norm2(h + ff_out)  # B x M x H
        if 'in' in self.mem_enc_type:
            out = self.memories['in'](h)
        out = self.norm2(h + out)  # B x M x H
        if 'after' in self.memories.keys():
            out = self.memories['after'](out)
        return out


class TransformerModelAdpSpn(nn.Module):
    # def __init__(self, vocab_size, hidden_size, nb_heads, nb_layers,
    #              attn_span, **kargs):
    def __init__(self, params, **kargs):
        nn.Module.__init__(self)
        # token embeddings

        # vocab_size = args.n_token
        vocab_size = params.n_words
        # nb_layers = args.dimensions[1]
        nb_layers = params.n_layers
        # d_model = args.dimensions[2]
        d_model = params.emb_dim
        hidden_size = d_model
        # inner_hid_size = args.dimensions[3]
        inner_hid_size = hidden_size * 4
        # nb_heads = args.dimensions[4]
        nb_heads = params.n_heads
        # dropout = args.dropout
        dropout = params.dropout
        # attn_span = args.attn_span
        attn_span = params.attn_span
        adapt_span_params = {}
        for item in params.adapt_span_params.split(','):
            k, v = item.split('=')
            if v == 'true' or v == 'false':
                v = np.bool(v)
            elif '0.' in v:
                v = np.float(v)
            else:
                v = np.int(v)
            adapt_span_params[k] = v
        # adapt_span_params = ast.literal_eval(params.adapt_span_params)

        self.vocab_size = vocab_size
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.nb_heads = nb_heads
        self.inner_hid_size = inner_hid_size
        self.attn_span = attn_span

        self.in_emb = nn.Embedding(vocab_size, hidden_size)

        # self.out_emb = nn.Linear(hidden_size, vocab_size)
        self.out_emb = PredLayer(params) # to use adaptive softmax

        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))
        
        mem_enc_positions_dict = dict()
        if params.use_memory:
            for k, v in params.mem_enc_positions:
                mem_enc_positions_dict.setdefault(k, []).append(v)

        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            if params.persistent_memory:
                self.layers.append(
                    TransformerSeqLayerPersistentMem(
                        hidden_size=hidden_size, nb_heads=nb_heads,
                        attn_span=attn_span, inner_hidden_size=inner_hid_size,
                        dropout=dropout, adapt_span_params=adapt_span_params, **kargs)
                )
            elif params.persistent_memory_pkm:
                self.layers.append(
                    TransformerSeqLayerPersistentMemPKM(
                        hidden_size=hidden_size, nb_heads=nb_heads,
                        attn_span=attn_span, inner_hidden_size=inner_hid_size,
                        dropout=dropout, adapt_span_params=adapt_span_params, **kargs)
                )
            elif i in mem_enc_positions_dict:
                self.layers.append(
                    TransformerSeqLayerPKM(
                        hidden_size=hidden_size, nb_heads=nb_heads,
                        attn_span=attn_span, inner_hidden_size=inner_hid_size,
                        dropout=dropout, adapt_span_params=adapt_span_params,
                        mem_enc_type=mem_enc_positions_dict[i],
                        params=params,
                        **kargs)
                )
            else:
                self.layers.append(
                    TransformerSeqLayer(
                        hidden_size=hidden_size, nb_heads=nb_heads,
                        attn_span=attn_span, inner_hidden_size=inner_hid_size,
                        dropout=dropout, adapt_span_params=adapt_span_params, **kargs)
                )

    def forward(self, x, y, h_cache, eval_only=False, loss_div=1):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        # out = F.log_softmax(self.out_emb(h), dim=-1)
        # scores = self.out_emb(h)
        # out = F.log_softmax(scores, dim=-1)
        # compute loss
        # out = out.view(-1, out.size(-1))
        # loss = torch.nn.functional.nll_loss(out, y.contiguous().view(-1))\
        scores, loss = self.out_emb(h.reshape(-1,h.shape[-1]), y.reshape(-1), get_scores=True)
        scores = scores.reshape(*h.shape[:-1], scores.shape[-1])

        # loss_value = loss.item() / loss_div

        # if not eval_only:
        #     # loss term from adaptive-span
        #     if self.layers[0].attn.attn.adapt_span_enabled:
        #         loss += sum(layer.attn.attn.adaptive_span.get_loss()
        #                     for layer in self.layers)


        # return loss / loss_div, loss_value, h_cache_next, scores
        return loss / loss_div, h_cache_next, scores

        # return out, h_cache_next
