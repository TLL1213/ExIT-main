import json
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse


def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, default='../data/fcgec/FCGEC_test_only_error_sampling_10_glm4_9B.jsonl', help='')
    parser.add_argument('--save_path', type=str, default='../data/fcgec/FCGEC_test_only_error_sampling_10_glm4_9B.jsonl', help='')
    parser.add_argument('--vocab_special_path', type=str, default='/_glm4/glm-4-9b-chat-fcgec-only-first-and-right', help='')


    return parser.parse_args()

if __name__ == '__main__':
    """
    补全数据中的token_id，无论是测试集、训练集还是验证集都需要经过这一步（填充的是自己模型词表的token_id）
    如果使用在qwen2.5-14B训练的模型，则执行完这一步之后，需要执行GLM4toQwen2.5  将token_id 转换为适应qwen2.5-14B词表的
    """
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    vocab_special_path = args.vocab_special_path



    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print('aa')
    tokenizer = AutoTokenizer.from_pretrained(vocab_special_path, trust_remote_code=True)
    # 加载词表
    vocab = tokenizer.get_vocab()
    # vocab = {k.decode('utf-8'): v for k, v in vocab.items()} # 转换类型为str
    # vocab = {}
    # for k, v in vocab_origin.items():
    #     try:
    #         vocab[k.decode('utf-8')] = v
    #     except:
    #         vocab[k] = v

    pbar = tqdm(total=len(data), desc='gain token id...')
    save_data = []
    # 获取真正的token_id并重新存储
    for item in data:
        sentence = item['sentence']
        sampling_responses = item['sampling_responses']
        for sampling_res in sampling_responses:
            response = sampling_res['response']
            response_logprob = sampling_res['response_logprob']
            for token in response_logprob:
                token_txt = token['token_txt']
                if token_txt != '': # 可能会出现空字符
                    try:
                        token_id = vocab[token_txt.encode('utf-8')]
                    except:
                        token_id = 3122 # 少数汉字表示进行的完整编码
                else:
                    token_id = 3122 # 设置成3122是为了兼容qwen2.5的词表，其中unk的tokenid就是3122.因为这里是为了验证模型在非同一个系列模型的泛化性，所以这里做了兼容
                token['token_id'] = token_id
        save_data.append(item)
        pbar.update(1)

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in save_data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行