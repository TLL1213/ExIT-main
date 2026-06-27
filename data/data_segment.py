import json
import random
import os
import pandas as pd
import argparse

# 切分每个列表为 K 份
def split_list(lst, k):
    n = len(lst)
    chunk_size = n // k
    remainder = n % k
    chunks = []
    start = 0
    for i in range(k):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def parse_args():
    # 注意现在微调的是哪个模型
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', type=str, default='./fcgec/right_FCGEC/FCGEC_train.csv', help="待切分训练集路径")
    parser.add_argument('--save_dir', type=str, default='./fcgec/segment', help="保存切分数据的文件夹路径")
    parser.add_argument('--K', type=int, default=4, help="切分为K份")
    parser.add_argument('--seed', type=int, default=2025, help="种子")

    return parser.parse_args()


if __name__ == '__main__':
    """训练数据太少，分布过于不均衡，这里切分数据集使用K折方法拓展训练数据"""

    args = parse_args()
    data_path = args.data_path
    data = pd.read_csv(data_path)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    K = args.K

    IWC = []
    CM = []
    CR = []
    SC = []
    IWO = []
    ILL = []
    AM = []
    RIGHT = []
    for idx, row in data.iterrows():
        sentence = row['Sentence']
        error_type = row['Type']
        error_type_list = error_type.split(';')
        right_sentence = row['Correction'].split('\t')[0].strip()
        if '*' in error_type_list: RIGHT.append((sentence, right_sentence))
        if 'IWC' in error_type_list: IWC.append((sentence, right_sentence))
        if 'CM' in error_type_list: CM.append((sentence, right_sentence))
        if 'CR' in error_type_list: CR.append((sentence, right_sentence))
        if 'SC'in error_type_list: SC.append((sentence, right_sentence))
        if 'IWO' in error_type_list: IWO.append((sentence, right_sentence))
        if 'ILL' in error_type_list: ILL.append((sentence, right_sentence))
        if 'AM' in error_type_list: AM.append((sentence, right_sentence))

    # 随机打乱每个列表
    random.seed(args.seed)
    random.shuffle(IWC)
    random.shuffle(CM)
    random.shuffle(CR)
    random.shuffle(SC)
    random.shuffle(IWO)
    random.shuffle(ILL)
    random.shuffle(AM)
    random.shuffle(RIGHT)
    IWC_chunks = split_list(IWC, K)
    CM_chunks = split_list(CM, K)
    CR_chunks = split_list(CR, K)
    SC_chunks = split_list(SC, K)
    IWO_chunks = split_list(IWO, K)
    ILL_chunks = split_list(ILL, K)
    AM_chunks = split_list(AM, K)
    RIGHT_chunks = split_list(RIGHT, K)

    section = []
    # 输出结果，可以将这些结果保存到文件中或进一步处理
    for i in range(K):
        print(f"Fold {i+1}:")
        print(f"IWC: {len(IWC_chunks[i])}")
        print(f"CM: {len(CM_chunks[i])}")
        print(f"CR: {len(CR_chunks[i])}")
        print(f"SC: {len(SC_chunks[i])}")
        print(f"IWO: {len(IWO_chunks[i])}")
        print(f"ILL: {len(ILL_chunks[i])}")
        print(f"AM: {len(AM_chunks[i])}")
        print(f"RIGHT: {len(RIGHT_chunks[i])}")
        print("-" * 40)
        section_i = []
        for item in IWC_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in CM_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in CR_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in SC_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in IWO_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in ILL_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in AM_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        for item in RIGHT_chunks[i]:
            section_i.append({
                "messages": [
                    {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
                    {"role": "user", "content": f"{item[0]}"},
                    {"role": "assistant", "content": f"{item[1]}"}
                ]
            })
        section.append(section_i)

    # 保存仅仅含有某个部分的数据
    for idx in range(len(section)):
        save_path = os.path.join(save_dir, f'FCGEC_train_section{idx}.jsonl')
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in section[idx]:
                json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
                f.write("\n")  # 每个 JSON 对象占一行

    section_k_1 = [] # 用于保存合并了k-1份数据的内容
    for idx in range(len(section)):
        temp = section[:idx] + section[idx + 1:]
        merged_list = []
        # 遍历每个列表并使用 extend() 方法将其元素添加到主列表
        for sublist in temp:
            merged_list.extend(sublist)
        section_k_1.append(merged_list)
    # 保存不含有某个部分的k-1个数据
    for idx in range(len(section_k_1)):
        save_path = os.path.join(save_dir, f'FCGEC_train_no_section{idx}.jsonl')
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in section_k_1[idx]:
                json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
                f.write("\n")  # 每个 JSON 对象占一行

