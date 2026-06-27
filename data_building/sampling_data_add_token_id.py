import json
import argparse

def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, default='../data/fcgec/FCGEC_valid_qwen2_7B_sampling_10.jsonl', help='')
    parser.add_argument('--save_path', type=str, default='../data/fcgec/FCGEC_valid_qwen2_7B_sampling_10.jsonl', help='')
    parser.add_argument('--vocab_path', type=str, default='../data/qwen2.5-14B/vocab.json', help='')
    parser.add_argument('--vocab_special_path', type=str, default='../data/qwen2.5-14B/tokenizer_config.json', help='')


    return parser.parse_args()

if __name__ == '__main__':
    """
    补全数据中的token_id，无论是测试集、训练集还是验证集都需要经过这一步
    """
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    vocab_path = args.vocab_path
    vocab_special_path = args.vocab_special_path



    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print('aa')
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

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in save_data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行