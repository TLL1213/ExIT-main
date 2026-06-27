import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, Embedding_size, K_size, V_size, n_head, drop_out=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = Embedding_size
        self.d_k = K_size
        self.d_v = V_size
        self.num_heads = n_head
        self.dropout = nn.Dropout(drop_out)

        # linear projections
        self.W_Q = nn.Linear(Embedding_size, K_size * n_head)
        self.W_K = nn.Linear(Embedding_size, K_size * n_head)
        self.W_V = nn.Linear(Embedding_size, V_size * n_head)
        self.W_out = nn.Linear(V_size * n_head, Embedding_size)

        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (Embedding_size + K_size)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (Embedding_size + K_size)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (Embedding_size + V_size)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (Embedding_size + V_size)))

    def forward(self, Q, K, V, attn_mask, **kwargs):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        # pre-process mask
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # broadcast
            attn_mask = attn_mask.bool()

        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)  # attention weights
        attns = self.dropout(attns)

        # calculate output
        output = torch.matmul(attns, V)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output