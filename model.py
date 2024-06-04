import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Note:

1. Model architecture heavily inspired by Andrej Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT) 
   and his tutorial on youtube (https://youtu.be/kCc8FmEb1nY?si=HsUz8gxVYrJF9n9o).

2. To ensure that I dont just copy paste the original code, I have understood the architecture at the lowest 
   level, wrote all the components down on paper, and then implemented them myself :)

Abbreviations used:
B      = batch size
T      = sequence length
N      = number of layers
N_embd = embedding size
N_head = number of heads
H_size = head size
V_size = vocab size

"""

class SelfAttention(nn.Module):
    """
    Input: normalized tensor (B, T, N_embd)

        Attention_params: Q, K, V --> (B, N_head, T, H_size)

        Attention: Softmax(Masking((Q @ K)/sqrt(size_K))) @ V

        Projection: (B, N_head, T, H_size) --> (B, T, N_embd) & N_embd = N_head * H_size

    Output: normalized tensor (B, T, N_embd)

    """
    def __init__(self, config):
        super().__init__()
        assert config.N_embd % config.N_head == 0
        self.emb_size = config.N_embd
        self.N_head = config.N_head
        self.H_size = config.N_embd // config.N_head
        self.qkv = nn.Linear(self.emb_size, self.emb_size * 3, bias=config.bias)
        self.proj = nn.Linear(self.emb_size, self.emb_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T = x.size()[0], x.size()[1]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # Q, K, V --> (B, N_head, T, H_size)
        q = q.view(B, T, self.N_head, self.H_size).transpose(1, 2)
        k = k.view(B, T, self.N_head, self.H_size).transpose(1, 2)
        v = v.view(B, T, self.N_head, self.H_size).transpose(1, 2)

        # (B, N_head, T, H_size) @ (B, N_head, H_size , T) --> (B, N_head, T, T)
        raw_scores = q @ k.transpose(-2, -1) / self.H_size**0.5
        raw_scores = F.softmax(raw_scores.masked_fill(torch.tril(raw_scores, diagonal=0) == 0, float('-inf')), dim=-1)
        raw_scores = self.attn_dropout(raw_scores)

        # (B, N_head, T, T) @ (B, N_head, T, H_size) --> (B, N_head, T, H_size)
        attention = raw_scores @ v
        attention = self.proj_dropout(self.proj(attention.transpose(1, 2).contiguous().view(B, T, self.emb_size)))
        return attention


class MLP(nn.Module):
    """
    Input: (B, T, N_embd)

        Linear Layer: (N_embd, dim)
                Activation
        Linear Layer: (dim, N_embd)
                Dropout

    Output: (B, T, N_embd)
    """
    def __init__(self, config):
        super().__init__()
        self.ffw1 = nn.Linear(config.N_embd, 4 * config.N_embd, bias=config.bias)
        self.activation = nn.ReLU()
        self.ffw2 = nn.Linear(4 * config.N_embd, config.N_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ffw1(x)
        x = self.activation(x)
        x = self.ffw2(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    """
    Input: (B, T, N_embd) --> embedded vector

        LayerNorm: (B, T, N_embd)
        Self Attention: (B, T, N_embd) --> (B, T, N_embd)
            Residual connection
        LayerNorm: (B, T, N_embd)
        MLP: (B, T, N_embd) --> (B, T, N_embd)
            Residual connection

    Output: (B, T, N_embd)
    """
    def __init__(self, config):
        super().__init__()
        self.B = config.batch_size
        self.T = config.context_length
        self.N_emb = config.N_embd
        self.ln1 = nn.LayerNorm(self.N_emb, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm( self.N_emb, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.B = config.batch_size
        self.T = config.context_length
        self.N_emb = config.N_embd
        self.V_size = config.vocab_size

        self.transformer = nn.ModuleDict(dict(
            token_embd = nn.Embedding(self.V_size, self.N_emb),
            position_embd = nn.Embedding(self.T, self.N_emb),
            dropout = nn.Dropout(config.dropout),
            attention_blocks = nn.ModuleList([Block(config) for _ in range(config.N_layer)]),
            ln = nn.LayerNorm(self.N_emb, bias=config.bias)
        ))

        self.vocab_proj = nn.Linear(self.N_emb, self.V_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x, targets=None):
        B, T, = x.size()
        device = x.device
        assert T <= self.T, f"Cannot pass sequence of length {T}, when the block size is {self.T}"
        # print(x.shape)
        tok_emb = self.transformer.token_embd(x)
        pos_emb = self.transformer.position_embd(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        for block in self.transformer.attention_blocks:
            x = block(x)

        x = self.transformer.ln(x)

        if targets is not None:
            logits = self.vocab_proj(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            logits = self.vocab_proj(x)
            loss = None

        return logits, loss