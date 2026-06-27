import json
from transformers import AutoTokenizer
from tqdm import tqdm

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    """
    因为glm4和qwen2.5系列模型词表不兼容，这里对候选句子进行映射
    将GLM4输出的候选句子的token，映射成为适应qwen2.5系列的
    """
    tokenizer_path = '../save_model/BeLLMSM_alpha0.4_CLS/best-model'
    # fcgec-test
    # input_file = './fcgec/FCGEC_test_only_error_sampling_10_glm4_9B.jsonl'
    # output_file = './fcgec/FCGEC_test_only_error_sampling_10_glm4_9B_2qwen2.5.jsonl'

    # NaCGEC
    input_file = './nacgec/NACGEC_sampling_10_glm4_9B.jsonl'
    output_file = './nacgec/NACGEC_sampling_10_glm4_9B.jsonl'

    # NaSGEC-exam
    # input_file = './nasgec/NASGEC_sampling_10_glm4_9B.jsonl'
    # output_file = './nasgec/NASGEC_sampling_10_glm4_9B.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data = load_data(input_file)
    num = 0
    pbar = tqdm(total=len(data), desc='map token id...')
    for idx, item in enumerate(data):
        sampling_responses = item['sampling_responses']
        for res_item in sampling_responses:
            response_logprob = res_item['response_logprob']
            for res_log in response_logprob:
                qwen_token_id = tokenizer(res_log['token_txt'])['input_ids']
                if len(qwen_token_id) == 1:
                    res_log['token_id'] = qwen_token_id[0]
                else:
                    num += 1
        pbar.update(1)

    print(f'映射失败数量:{num}')
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行