import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer


class BeliefTrainer(Trainer):
    def __init__(self, *args, alpha=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # 真实标签 y 的权重
        self.beta = 1 - alpha  # 次优得分 s 的权重
        # 在初始化时定义损失函数对象，只创建一次
        # 根据你 BeLLMSM 模型中是否应用了 Sigmoid，来选择合适的损失函数
        # 如果 BeLLMSM 最终输出的是 0-1 范围的概率值 (经过 Sigmoid)，则使用 MSELoss
        self.loss_fct = nn.MSELoss() # 回归 （若预测概率则这一个）
        # self.loss_fct = nn.BCEWithLogitsLoss() # 分类 （若label的格式是分类任务则这个）

        # 如果 BeLLMSM 最终输出的是原始 logits (未经过 Sigmoid)，且你希望它是概率型输出，则使用 BCEWithLogitsLoss
        # self.loss_fct = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # 解包输入数据
        labels = inputs.pop("labels")
        reference_score = inputs.pop("ref_scores")
        # 前向传播
        predicted_scores = model(**inputs)

        if predicted_scores is None:
            raise ValueError("Model output does not contain 'predicted_scores' or 'raw_predicted_scores'. "
                             "Please ensure BeLLMSM.forward() returns it.")
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.float32).to(
                predicted_scores.device if predicted_scores is not None else predicted_scores.device)
            reference_score = torch.tensor(reference_score, dtype=torch.float32).to(
                predicted_scores.device if predicted_scores is not None else predicted_scores.device)

        else:
            labels = labels.to(dtype=torch.float32).to(predicted_scores.device if predicted_scores is not None else predicted_scores.device)
            reference_score = reference_score.to(dtype=torch.float32).to(predicted_scores.device if predicted_scores is not None else predicted_scores.device)

        # 确保预测得分和标签的形状匹配
        if predicted_scores is not None and predicted_scores.shape != labels.shape:
            raise ValueError(f"Shape mismatch: predicted score {predicted_scores.shape} "
                             f"Mismatch with label {labels.shape}")

        # 计算自定义损失
        loss_main = self.loss_fct(predicted_scores, labels)  # predicted_scores 应该已经介于 0-1
        loss_aux = self.loss_fct(predicted_scores, reference_score)  # predicted_scores 应该已经介于 0-1

        loss = self.alpha * loss_main + self.beta * loss_aux
        # loss = loss_main

        if 'loss' not in locals():  # 如果上面没有计算出 loss，说明逻辑有问题
            raise RuntimeError("Loss calculation failed. Check your model output and chosen loss function.")

        # 6. 返回损失和模型输出
        return (loss, predicted_scores) if return_outputs else loss