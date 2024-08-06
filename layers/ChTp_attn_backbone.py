from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

class model_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None, channel_proj_len=128,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, fusion_dropout=0, proj_dropout=0, channel_attn_type="parallel", **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
            
        # Patching
        self.patch_len = patch_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        channel_proj_len = channel_proj_len
        # Backbone 
        self.backbone = iEncoder(c_in, patch_num=patch_num, channel_proj_len=channel_proj_len, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fusion_dropout=fusion_dropout, proj_dropout=proj_dropout, channel_attn_type=channel_attn_type, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        # z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class iEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, channel_proj_len=128, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, pre_norm=False,
                 pe='zeros', learn_pe=True, fusion_dropout=0, proj_dropout=0, channel_attn_type="parallel", **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.c_in = c_in
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = Encoder(c_in, q_len, d_model, n_heads, channel_proj_len=channel_proj_len,d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn, fusion_dropout=fusion_dropout, proj_dropout=proj_dropout, channel_attn_type=channel_attn_type)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = self.dropout(x + self.W_pos)                                         # u: [bs x nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs x nvars x patch_num x d_model]

        
        return z    


class Encoder(nn.Module):
    def __init__(self, c_in, q_len, d_model, n_heads, channel_proj_len = 128,d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        n_layers=1, pre_norm=False, store_attn=False, fusion_dropout=0, proj_dropout=0, channel_attn_type='parallel'):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(c_in, q_len, d_model,channel_proj_len=channel_proj_len, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, pre_norm=pre_norm, 
                                                      store_attn=store_attn, fusion_dropout=fusion_dropout, proj_dropout=proj_dropout, channel_attn_type=channel_attn_type) for i in range(n_layers)])

    def forward(self, src:Tensor):
        output = src
        for mod in self.layers: output = mod(output)
        return output
        

class EncoderLayer(nn.Module):  #i means channel-independent
    def __init__(self, c_in, q_len, d_model, channel_proj_len, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", pre_norm=False, fusion_dropout=0, proj_dropout=0, channel_attn_type='parallel'):
        super().__init__()
        self.c_in = c_in
        self.channel_attn = ChannelAttentionLayer(q_len, d_model, channel_proj_len, dropout=dropout, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.temporal_attn = TemporalAttentionLayer(d_model, dropout=dropout, n_heads=n_heads, attn_dropout = attn_dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias)) 
        self.dropout_attn_sublayer1 = nn.Dropout(fusion_dropout)
        self.dropout_attn_sublayer2 = nn.Dropout(fusion_dropout)
        self.channel_attn_type = channel_attn_type
        if "batch" in norm.lower():
            # self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_attn_sublayer1 = nn.BatchNorm2d(c_in)
            self.norm_attn_sublayer2 = nn.BatchNorm2d(c_in)
        else:
            self.norm_attn_sublayer1 = nn.LayerNorm(d_model)
            self.norm_attn_sublayer2 = nn.LayerNorm(d_model)

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            # self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_ffn = nn.BatchNorm2d(c_in)
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_num x d_model]
        if self.channel_attn_type == "parallel":
            x_temporal = self.temporal_attn(x)                                       # x_temporal: [bs x nvars x patch_num x d_model]
            x_channel = self.channel_attn(x)                                         # x_channel: [bs x nvars x patch_num x d_model]
            x = self.dropout_attn_sublayer1(x_temporal) + self.dropout_attn_sublayer2(x_channel) + x

            if not self.pre_norm:	
                x = self.norm_attn_sublayer1(x)
    
            # Feed-forward sublayer
            if self.pre_norm:
                x = self.norm_ffn(x)
    
            x2 = self.ff(x)
            ## Add & Norm
            x = x + self.dropout_ffn(x2) # Add: residual connection with residual dropout
            if not self.pre_norm:
                x = self.norm_ffn(x)
            
        elif self.channel_attn_type == "sequential":
            x_temporal = self.temporal_attn(x)                                       # x_temporal: [bs x nvars x patch_num x d_model]
            x = x + self.dropout_attn_sublayer1(x_temporal)
            x = self.norm_attn_sublayer1(x)
            x_channel = self.channel_attn(x)                                         # x_channel: [bs x nvars x patch_num x d_model]
            x = x + self.dropout_attn_sublayer2(x_channel)
            x = self.norm_attn_sublayer2(x)
            x2 = self.ff(x)
            ## Add & Norm
            x = x + self.dropout_ffn(x2) # Add: residual connection with residual dropout
            if not self.pre_norm:
                x = self.norm_ffn(x)
        return x    


class ChannelAttentionLayer(nn.Module):
    def __init__(self, seq_len, d_model, proj_len, dropout, n_heads, d_k = None, d_v = None, attn_dropout = 0, proj_dropout=0):
        super().__init__()
        self.W_P = nn.Linear(seq_len*d_model, proj_len)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.P_W = nn.Linear(proj_len, seq_len*d_model)        # projection back to patch len
        self.dropout = nn.Dropout(proj_dropout)

        self.self_attn = _MultiheadAttention(proj_len, 4, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)
    
    def forward(self, x) -> Tensor: 
        patch_num = x.shape[2]                                                              # x: [bs x nvars x patch_num x d_model]       
        x = torch.reshape(x, (x.shape[0], x.shape[1],x.shape[2]*x.shape[3]))                 # x: [bs x nvars x patch_num * d_model]
        x = self.dropout(self.W_P(x))                                                       # x: [bs x nvars x proj_len]
        x = self.self_attn(x)                                                               # x: [bs x nvars x proj_len]
        x = self.dropout(self.P_W(x))                                                       # x: [bs x nvars x patch_num x d_model]
        x = torch.reshape(x, (x.shape[0], x.shape[1], patch_num, -1))                       # x: [bs x nvars x patch_num x d_model]
        return x



class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model, dropout, n_heads, d_k = None, d_v = None, attn_dropout = 0):
        super().__init__()
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)
    
    def forward(self, x) -> Tensor: 
        n_vars = x.shape[1]                                                                 # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))                 # u: [bs * nvars x patch_num x d_model]

        z = self.self_attn(u)                                                               # z: [bs * nvars x patch_num x d_model]

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                           # z: [bs x nvars x patch_num x d_model]
        return z



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.attn_dropout = attn_dropout
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=self.attn_dropout, lsa=lsa)
        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # print("Q shape: ", Q.shape)
        # print("W_Q shape: ", self.W_Q.weight.shape)
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)

        # Apply Scaled Dot-Product Attention (multiple heads)

        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        return output


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights
