import torch
from torch import nn
from .EncoderLayer import Encoder
from transformers import PreTrainedModel, PretrainedConfig


def get_attn_mask_pad(seq, pad_id):
    """为序列生成注意力掩码（attention mask），用于在注意力机制中屏蔽掉填充（padding）的部分"""
    batch_size, len_data = seq.size()
    attn_pad = seq.data.eq(pad_id).unsqueeze(1)
    attn_pad = attn_pad.expand(batch_size, len_data, len_data)
    return attn_pad

class BeLLMSMConfig(PretrainedConfig):
    # 必须指定 model_type，通常是模型名称的小写形式
    model_type = "bellmsm"

    def __init__(self, vocab_size=151646, n_layers=6, Embedding_size=512, Q_size=64, K_size=64, V_size=64, n_head=8, FF_dimension=2048, length_tgv=10, pad_id=151643, max_source_length=256, max_candidate_length=1024, drop_out=0.1, **kwargs):
        super().__init__(**kwargs)
        # 将 BeLLMSM 模型的所有关键参数添加到这里
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.Embedding_size = Embedding_size
        self.Q_size = Q_size
        self.K_size = K_size
        self.V_size = V_size
        self.n_head = n_head
        self.FF_dimension = FF_dimension
        self.length_tgv = length_tgv
        self.pad_id = pad_id
        self.max_source_length = max_source_length
        self.max_candidate_length = max_candidate_length
        self.drop_out = drop_out


# Belief_LLM_select_Model
class BeLLMSM(PreTrainedModel):
    config_class = BeLLMSMConfig  # 关联自定义的配置文件
    base_model_prefix = "bellmsm"  # 用于在保存时区分模型文件，通常与 model_type 一致

    # def __init__(self, vocab_size, n_layers, Embedding_size, Q_size, K_size, V_size, n_head, FF_dimension, length_tgv, pad_id):
    def __init__(self, config: BeLLMSMConfig):
        super(BeLLMSM, self).__init__(config) # 传入config
        self.config = config  # 保存配置，方便后续访问
        self.embeddingsize = config.Embedding_size
        self.embed = nn.Embedding(config.vocab_size, config.Embedding_size, padding_idx=config.pad_id)  # 填充字符过多的情况下，不设置padding_idx可能会导致模型学习不到有用特征
        self.encoder = Encoder(config.max_candidate_length, config.Embedding_size, config.n_layers, config.n_head, config.FF_dimension, config.drop_out)
        self.regression_head = nn.Linear(config.Embedding_size, 1)  # output_dim通常为1表示一个分数
        self.pad_id = config.pad_id
        # 创建 Sigmoid 激活函数对象
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, logprobs, labels=None): # 这里的传参直接影响到使用trainer进行训练时，传入compute_loss函数的inputs的参数
        Encoder_input = self.embed(input_ids)
        mask = get_attn_mask_pad(input_ids, self.pad_id)
        Encoder_output = self.encoder(Encoder_input, logprobs, mask=mask)
        pooled_output = Encoder_output[:, 0, :]
        predicted_scores = self.regression_head(pooled_output)

        return predicted_scores
