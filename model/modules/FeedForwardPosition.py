from torch import nn
import torch.nn.functional as F

class FeedForwardPosition(nn.Module):
    def __init__(self, Embedding_size, FF_dimension):
        super(FeedForwardPosition, self).__init__()
        self.conv1 = nn.Conv1d(Embedding_size,FF_dimension, kernel_size = 1)
        self.conv2 = nn.Conv1d(FF_dimension,Embedding_size, kernel_size = 1)
        self.norm = nn.LayerNorm(Embedding_size)
        self.activation = nn.ReLU()  # 或者 nn.GELU()、nn.ReLU()
    def forward(self,attn_score):
        temp = attn_score
        output= self.conv1(attn_score.transpose(1,2))
        output = self.activation(output)  # 添加激活函数
        output = self.conv2(output).transpose(1,2)
        # 老版本transformers的设计
        # output = self.norm(output + temp)
        # Pre-LN设计
        output = self.norm(output)
        output = output + temp
        return output

class FeedForwardPosition(nn.Module):
    def __init__(self, Embedding_size, FF_dimension, drop_out=0.1):
        super(FeedForwardPosition, self).__init__()
        self.d_model = Embedding_size
        self.d_ff = FF_dimension
        self.conv1 = nn.Conv1d(Embedding_size, FF_dimension, 1, 1, 0)
        self.conv2 = nn.Conv1d(FF_dimension, Embedding_size, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out
