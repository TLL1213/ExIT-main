from typing import Dict, List
from transformers import (
    DataCollatorWithPadding,
)
import torch


# 自定义 DataCollator，不生成 attention_mask
class BeliefDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        #transformers.DataCollatorWithPadding在处理数据时，默认情况下会移除它不认识或者不需要用于模型输入的键。这里需要手动加入
        # print('*'*100)
        # ref_scores = [feature.pop('ref_scores') for feature in features]# 备份非标准键的数据
        batch = super().__call__(features)
        # batch["ref_scores"] = torch.tensor(ref_scores, dtype=torch.float)
        # 目的是删除生成的attention_mask造成的影响
        if "attention_mask" in batch:
            del batch["attention_mask"]
        return batch