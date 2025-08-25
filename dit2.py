import math

import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

if torch.cuda.is_available():
    import flash_attn
    import flash_attn.layers.rotary


"""
╭ CONVENTIONS ───────────────────────────────────────────────────────────────────╮
│ ├─• B        ▶ batch size                                                      │
│ ├─• T        ▶ number of tokens in a batch i.e. length of a sequence/sentence  │
│ ├─• C        ▶ embedding dimension of each token                               │
│ │                                                                              │
│ ├─• H        ▶ number of heads                                                 │
│ ├─• V        ▶ vocabulary size                                                 │
│ │                                                                              │
│ ├─• cond_dim ▶ output size of the TimestepEmbedding layer                      │
│ ╰─• f_dim    ▶ initial embedding size for of the frequency                     │
╰────────────────────────────────────────────────────────────────────────────────╯
"""




# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                               Rotary  PE                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class Rotary(torch.nn.Module):
    def __init__(self, c, base=10_000):
        super().__init__()
        dtype = torch.get_default_dtype()
        inv_freq = 1. / (base ** (torch.arange(0, c, 2, dtype=dtype) / c))
        self.register_buffer('inv_freq', inv_freq)
        
        self.T_cached = 0                                                   # we will store the cos and sin values for the max T yet

    
    def forward(self, x):
        T = x.shape[1]
        dtype = self.inv_freq.dtype

        if self.T_cached < T:
            t = torch.arange(T, dtype=dtype, device=x.device)               # T
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())       # T c//2, first row is t[0]*inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)                         # T c

            self.cos = repeat(emb.cos(), 'T c -> 1 T 3 1 c')                # 1 T 3 1 c
            self.sin = repeat(emb.sin(), 'T c -> 1 T 3 1 c')                # 1 T 3 1 c
            
            self.cos[:,:,2,:,:].fill_(1.)                                   # ◀─┬ This makes the transformation 
            self.sin[:,:,2,:,:].fill_(0.)                                   # ◀─╯ on values an identity
            
            self.T_cached = T                                               # update T_cached

        cos = self.cos[:, :T, :, :, :]                                      # ◀─┬ cut based on the 
        sin = self.sin[:, :T, :, :, :]                                      # ◀─╯ token length

        return cos, sin



def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                            Embedding Layers                                  │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, cond_dim, f_dim=256, max_period = 10_000):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(f_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.f_dim = f_dim

        half = self.f_dim // 2
        arange = torch.arange(0, half, dtype=torch.get_default_dtype())       # f_dim//2
        freqs = torch.exp(- math.log(max_period)* arange / half )             # f_dim//2
        self.register_buffer('freqs', freqs)  


    def forward(self, t):
        args = t[:, None].float() * self.freqs[None]                          # B f_dim//2
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)     # B (f_dim//2)*2 != B f_dim if f_dim is odd
        embedding = F.pad(embedding, (0, self.f_dim % 2))                     # B f_dim

        return self.FFN(embedding)                                            # B cond_dim



class EmbeddingLayer(nn.Module):
    def __init__(self, C, V):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((V, C)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):                                                     # B T                                              
        return self.embedding[x]                                              # B T C





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                               Layer Norm                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class LayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([C]))
        self.bias  = nn.Parameter(torch.zeros([C]))
        self.C     = C
    
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.C])
        return x * self.scale[None, None, :] + self.bias[None, None, :]





# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                           Multi Head Attention                               │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class MultiHeadAttention(nn.Module):
    def __init__(self, C: int = 256, H: int = 8, p_dropout: float = 0.1):
        super().__init__()
        assert C % H == 0, "embedding dimension C must be divisible by the number of heads H"
        
        self.C, self.H = C, H
        self.c         = int(C // H)

        self.W_qkv     = nn.Linear(C, 3 * C, bias=False)
        self.W_o       = nn.Linear(C, C) 
        self.dropout   = nn.Dropout(p_dropout)

        cuda = torch.cuda.is_available()
        self.attention = self._attention_cuda if cuda else self._attention
        

    def forward(self, x, rotary_cos_sin, seqlens):
        """ 
        x:              B, T C, tensor input
        rotary_cos_sin: cosine and sine tensor from Rotary class
        seqlens:        B T, how long is each sequence 
        """
        qkv = self.W_qkv(x)                                           # B T (three C)
        qkv = rearrange(qkv, 'B T (three H c) -> B T three H c',      # B T three H c
                        three=3, H=self.H)
        
        x = self.attention(qkv, rotary_cos_sin, seqlens)              # B T C 
        x = self.dropout(self.W_o(x))                                 # B T C 

        return x    


    def _attention(self, qkv, rotary_cos_sin, seqlens):
        cos, sin = rotary_cos_sin                                     #  ╭ rotary positional embedding 
        qkv = qkv * cos + rotate_half(qkv) * sin                      # ◀╯ B T three H c  

        qkv = rearrange(qkv, 'B T three H c -> B three H T c')
        q, k, v = qkv.unbind(dim=1)                                   # B three H S c -> 3 * B H T c

        c = q.shape[-1]                                               #  ╭ compute attention
        attn_scores = (q @ k.transpose(-2, -1)) * (c ** -0.5)         # ◀┤ B H T T
        attn_probs = F.softmax(attn_scores, dim=-1)                   # ◀┤ B H T T
        x = attn_probs @ v                                            # ◀╯ B H T c

        return rearrange(x, 'B H T c -> B T (H c)')                   # B T C


    def _attention_cuda(self, qkv, rotary_cos_sin, seqlens):
        B, T, _, H, c = qkv.shape
        dv            = qkv.device

        with torch.cuda.amp.autocast(enabled=False):                              #  ╭ rotary positional embedding           
            cos, sin = rotary_cos_sin                                             #  │
            cos = cos[0, :, 0, 0, :cos.shape[-1]//2].to(qkv.dtype)                # ◀┤  T c//2
            sin = sin[0, :, 0, 0, :sin.shape[-1]//2].to(qkv.dtype)                # ◀┤  T c//2
            qkv = flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)   # ◀╯  B T 3 H c
        
        qkv = rearrange(qkv, 'B T ... -> (B T) ...')                              # (B T) 3 H c
        cu_seqlens = seqlens.cumsum(-1) if seqlens else self.cu_seqlens(B, T, dv) # ◀─ B + 1, compute the cumulative sequence length

        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(     #  ╭ compute attention 
            qkv, cu_seqlens, T, 0., causal=False)                                 # ◀╯ (B T) 3 H c
        
        return rearrange(x, '(B T) H c -> B T (H c)', B=B)                        # B T C
        

    def cu_seqlens(self, B, T, device):
        return torch.arange(0, (B + 1)*T, T, dtype=torch.int32, device=device)
        




# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                           Feed Forward Network                               │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class FeedForward(nn.Module):
    def __init__(self, C: int = 64, factor: int = 4):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(C, factor*C),
            nn.GELU(approximate='tanh'),
            nn.Linear(factor*C, C)
        )
    
    def forward(self, x):
        return self.FFN(x)





# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                   Blocks                                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def add_scale(x: torch.Tensor,
              shift: torch.Tensor, 
              scale: torch.Tensor
              ) -> torch.Tensor:
    return x * (1 + scale) + shift



class DiTBlock(nn.Module):
    def __init__(self, C, H, cond_dim, p_dropout=0.1, FFN_ratio=4):
        super().__init__()
        self.H         = H

        self.norm1     = LayerNorm(C)
        self.attention = MultiHeadAttention(C, H)

        self.norm2     = LayerNorm(C)
        self.FFN       = FeedForward(C, FFN_ratio)
        self.dropout   = nn.Dropout(p_dropout)

        self.ALN       = nn.Linear(cond_dim, 6 * C)   # ◀╮ Adaptive Layer Normalization, used
        self.ALN.weight.data.zero_()                  #  │ for conditioning. Initialized at
        self.ALN.bias.data.zero_()                    #  ╰ zero


    def forward(self, x, rotary_cos_sin, conditioning, seqlens=None):
        (shift_att, scale_att, gate_att, 
         shift_FNN, scale_FNN, gate_FNN) = self.ALN(conditioning)[:, None].chunk(6, dim=2)

        x_as = add_scale(self.norm1(x), shift_att, scale_att)
        x = x + gate_att*self.attention(x_as, rotary_cos_sin, seqlens)
        
        x = x + gate_FNN*self.dropout(self.FFN(add_scale(self.norm2(x), shift_FNN, scale_FNN)))

        return x



class DiTLastBlock(nn.Module):
    def __init__(self, C, V, cond_dim):
        super().__init__()
        self.norm   = LayerNorm(C)

        self.linear = nn.Linear(C, V)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.ALN    = nn.Linear(cond_dim, 2 * C)
        self.ALN.weight.data.zero_()
        self.ALN.bias.data.zero_()


    def forward(self, x, conditioning):
        shift, scale = self.ALN(conditioning)[:, None].chunk(2, dim=2)

        x = add_scale(self.norm(x), shift, scale) 
        x = self.linear(x)
        return x
  




# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                                     DIT                                      │
# ╰──────────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DIT(nn.Module):                     #huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, 
                V: int,                   # ◀ vocabulary size
                C: int = 128,             # ◀ embedding dimension
                H: int = 4,               # ◀ number of heads
                cond_dim: int = 32,       # ◀ internal dimension for conditioning
                N: int = 3,               # ◀ number of blocks
                p: float = 0.1            # ◀ probability of dropout
                ):
        super().__init__()

        self.embedding = EmbeddingLayer(C, V)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary    = Rotary(C // H)
       
        blocks         = [DiTBlock(C, H, cond_dim, p) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        self.output    = DiTLastBlock(C, V, cond_dim)


    def forward(self, indices, sigma):
        x = self.embedding(indices)

        conditioning = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, conditioning, seqlens=None)

            x = self.output(x, conditioning)

        return x