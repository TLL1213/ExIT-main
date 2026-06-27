import json
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def read_sampling_data(data_path):
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print('aa')
    return data

def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, default='../data/fcgec/FCGEC_valid_qwen2_7B_sampling_10.jsonl', help='')
    parser.add_argument('--data_right_path', type=str, default='../data/fcgec/right_FCGEC/FCGEC_valid.csv', help='')
    parser.add_argument('--save_path', type=str, default='../data/fcgec/FCGEC_valid_qwen2_7B_sampling_10_label.jsonl', help='')



    return parser.parse_args()

if __name__ == '__main__':
    """
    获得label，并且修正数据格式（最终输入）   获得得分数据  只有训练集和验证集需要经过这一步   经过gain_fcgec_train_data_merge_segment.py的训练集已经有label了，所以这里仅仅为验证集补充即可
    """
    args = parse_args()
    data_path = args.data_path
    data_right_path = args.data_right_path
    save_path = args.save_path



    # 创建TF-IDF模型
    vectorizer = TfidfVectorizer()
    data_right = pd.read_csv(data_right_path)
    data = read_sampling_data(data_path)

    # 获得每个回答与label的得分（取最大值）
    save_data = []
    pbar = tqdm(total=len(data), desc='gain score...')
    for data_idx, item in enumerate(data):
        # 获取错误句子
        sentence = item['sentence']
        # 获取label
        labels = data_right.iloc[data_idx]['Correction'].split('\t')
        labels[-1] = labels[-1].strip()
        # 获取response
        sampling_responses = item['sampling_responses']
        for sampling_res in sampling_responses:
            response = sampling_res['response']
            max_cos = -1.0
            for g_s in labels:
                tfidf_matrix = vectorizer.fit_transform([response, g_s])
                # 计算余弦相似度
                cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                # 最好
                if max_cos < cos_sim[0][0]:
                    max_cos = cos_sim[0][0]

            sampling_res['score'] = max_cos
        save_data.append(item)
        pbar.update(1)

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in save_data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行



