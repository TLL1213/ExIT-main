import torch
from torch import nn
from .modules.MultiHeadAttention import MultiHeadAttention
from .modules.FeedForwardPosition import FeedForwardPosition
from .modules.PositionalEncoding import PositionalEncoding



class EncoderLayer(nn.Module):
    # def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
    def __init__(self, Embedding_size, n_head, FF_dimension, drop_out=0.1):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert Embedding_size % n_head == 0
        hdim = Embedding_size // n_head     # dimension of each attention head
        super(EncoderLayer, self).__init__()
        # LayerNorm
        self.norm1 = nn.LayerNorm(Embedding_size)
        self.norm2 = nn.LayerNorm(Embedding_size)
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(Embedding_size, hdim, hdim, n_head, drop_out)
        # Position-wise Feedforward Neural Network
        self.poswise_ffn = FeedForwardPosition(Embedding_size, FF_dimension, drop_out)

    def forward(self, enc_in, attn_mask):
        # reserve original input for later residual connections
        residual = enc_in
        # MultiHeadAttention forward
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        # residual connection and norm
        out = self.norm1(residual + context)
        residual = out
        # position-wise feedforward
        out = self.poswise_ffn(out)
        # residual connection and norm
        out = self.norm2(residual + out)

        return out


class Encoder(nn.Module):
    def __init__(
            # self, dropout_emb, num_layers, enc_dim, num_heads, dff, tgt_len,
            self, max_candidate_length, Embedding_size, n_layers, n_head, FF_dimension, drop_out
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        # The maximum length of input sequence
        self.tgt_len = max_candidate_length
        self.pos_emb = nn.Embedding.from_pretrained(PositionalEncoding(max_candidate_length, Embedding_size), freeze=True)
        self.emb_dropout = nn.Dropout(drop_out)
        self.layers = nn.ModuleList(
            [EncoderLayer(Embedding_size, n_head, FF_dimension, drop_out) for _ in range(n_layers)]
        )

    def forward(self, input_ids, logprobs, mask=None):
        batch_size, seq_len, d_model = input_ids.shape
        logprobs = logprobs.view(logprobs.size(0), -1)
        logprobs = torch.exp(logprobs)
        logprobs = torch.clamp(logprobs, min=1e-8)  # 添加一个小的 epsilon
        logprobs = logprobs.unsqueeze(2).expand(-1, -1, d_model)

        # add position embedding

        out = input_ids + self.pos_emb(torch.arange(seq_len, device=input_ids.device))  # (batch_size, seq_len, d_model)
        out = out * logprobs
        out = self.emb_dropout(out)
        # encoder layers
        for layer in self.layers:
            out = layer(out, mask)
        return out