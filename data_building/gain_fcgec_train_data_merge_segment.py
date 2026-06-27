import argparse
import json
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_token_id(vocab_path, vocab_special_path, data):
    """给数据补充token_id"""
    # 加载词表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 加载特殊字符词表
    with open(vocab_special_path, 'r', encoding='utf-8') as f:
        vocab_tmp = json.load(f)
        vocab_tmp = vocab_tmp['added_tokens_decoder']
        vocab_special = {}
        for k, v in vocab_tmp.items():
            vocab_special[v['content']] = k
    save_data = []
    pbar = tqdm(total=len(data), desc='gain token_id...')
    # 获取真正的token_id并重新存储
    for item in data:
        sentence = item['sentence']
        sampling_responses = item['sampling_responses']
        for sampling_res in sampling_responses:
            response = sampling_res['response']
            response_logprob = sampling_res['response_logprob']
            for token in response_logprob:
                token_txt = token['token_txt']
                try:
                    token_id = vocab[token_txt]
                except:
                    token_id = vocab_special[token_txt]
                token['token_id'] = token_id
        save_data.append(item)
        pbar.update(1)
    return save_data

def add_label(data_right_path, data):
    """计算数据的label"""
    # 创建正确修正的映射表(因为经过k-折得到的训练数据无法再和原本的存储的数据按索引一一对应了)
    data_right = pd.read_csv(data_right_path)
    data_right_dict = {}
    for idx, row in data_right.iterrows():
        sentence = row['Sentence']
        labels = row['Correction'].split('\t')
        labels[-1] = labels[-1].strip()
        data_right_dict[sentence] =labels
    # 开始计算得分
    # 创建TF-IDF模型
    vectorizer = TfidfVectorizer()
    save_data = []
    pbar = tqdm(total=len(data), desc='gain score...')
    for data_idx, item in enumerate(data):
        # 获取错误句子
        sentence = item['sentence']
        # 获取label
        labels = data_right_dict[sentence]
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
    return save_data

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='')
    # 根据使用模型的不同进行替换
    # data_path参数是之前通过K个模型交叉推理K次得到的推理结果
    parser.add_argument(
        '--data_path',
        type=str,
        nargs='+',
        default=[
            '../data/fcgec/segment/FCGEC_train_section0_sampling_10.jsonl',
            '../data/fcgec/segment/FCGEC_train_section1_sampling_10.jsonl',
            '../data/fcgec/segment/FCGEC_train_section2_sampling_10.jsonl',
            '../data/fcgec/segment/FCGEC_train_section3_sampling_10.jsonl',
        ],
        help=''
    )
    parser.add_argument('--save_path', type=str, default='../data/fcgec/segment/FCGEC_train_sampling_10.jsonl', help='')
    parser.add_argument('--vocab_path', type=str, default='../data/qwen2.5-14B/vocab.json', help='换成对应模型的词表')
    parser.add_argument('--vocab_special_path', type=str, default='../data/qwen2.5-14B/tokenizer_config.json', help='换成对应模型的特殊词表')
    parser.add_argument('--data_right_path', type=str, default='../data/fcgec/right_FCGEC/FCGEC_train.csv', help='')


    return parser.parse_args()

if __name__ == '__main__':
    """合并fcgec的训练数据，同时补充token_id和label"""
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    vocab_path = args.vocab_path
    vocab_special_path = args.vocab_special_path
    data_right_path = args.data_right_path




    data = []
    for dp in data_path:
        data_i = load_data(dp)
        data += data_i


    data = add_token_id(vocab_path, vocab_special_path, data)
    data = add_label(data_right_path, data)

    # 保存答案
    pbar = tqdm(total=len(data), desc='save data...')
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行
            pbar.update(1)