import argparse
import json
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

def remove_newline(data):
    """GLM-4在输出的时候会额外在首token带有一个换行，这里消除换行的影响"""
    pbar = tqdm(total=len(data), desc='Correct the result of glm-4...')
    for item in data:
        # uuid = item[1]
        # sentence = item[2]
        # sampling_res = []
        choices = item[3]
        for choice in choices:
            res = choice.message.content  # 采样的其中一个结果
            choice.message.content = res[1:] # 删除返回候选纠正中的第一个换行。
            # 遍历某个修正结果的每个token
            del choice.logprobs.content[0] # 删除首token的换行
        pbar.update(1)
    return data

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup    tables between utf-8 bytes and unicode strings.    """
    bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_reverse_map(forward_map):
    """
    根据正向映射生成反向映射
    返回字典：{unicode_char: byte_value}
    """
    return {v: k for k, v in forward_map.items()}


def bytes_to_unicode_str(byte_sequence):
    return ''.join([forward_map[b] for b in byte_sequence])


def unicode_str_to_bytes(unicode_str):
    return bytes([reverse_map[c] for c in unicode_str])


# 生成映射表
forward_map = bytes_to_unicode()
reverse_map = get_reverse_map(forward_map)

# 将单个token进行解码
def decode_token(token_txt):
    """若是qwen系列的vocab.json，想要看到词表中的词语的简体中文形式，需要解码"""
    token_txt_utf8 = unicode_str_to_bytes(token_txt)
    try:
        recovered_text = token_txt_utf8.decode('utf-8')
    except  UnicodeDecodeError:
        recovered_text = token_txt
    return recovered_text

def query(input_id, uuid, error_sentence):
    """访问openai接口的函数"""
    # system设置风格
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
            {"role": "user", "content": f"{error_sentence}"},
            # {"role": "user", "content": f"{few_shot_prompt}{error_sentence}"},
        ],
        model="qwen2.5",
        # model="lora1",
        n=10,
        logprobs=True,
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # 不是qwen3的话注销
    )

    output = chat_completion.choices

    return input_id, uuid, error_sentence, output

def batch_inference(data, batch_size=5):
    """批推理，大幅度减少推理时间"""
    pbar = tqdm(total=len(data), desc='batch data inference.')
    inference_res_list = []
    for i in range(0, len(data), batch_size):
        if i + batch_size < len(data):
            batch = data[i:i + batch_size]
        else:
            batch = data[i:]
        batch_res = []  # batch_res 为batch中每个元素的推理结果，与下标索引一一对应
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(query, input_id=idx, uuid=tmp[0], error_sentence=tmp[1]) for idx, tmp in enumerate(batch)
            ]
            for future in concurrent.futures.as_completed(futures):
                # future.result()
                input_id, uuid, error_sentence, res = future.result()  # 获取返回值
                batch_res.append((input_id, uuid, error_sentence, res))  # 将返回值存储到列表中
        # 因为并发的缘故，这里的input_id不是递增的，所以重新按照input_id排序
        batch_res = sorted(batch_res, key=lambda x: x[0])
        inference_res_list += batch_res
        if i + batch_size < len(data):
            pbar.update(batch_size)
        else:
            pbar.update(len(data) - i)
    return inference_res_list


def load_data(data_path):
    sentences = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split('\t')
            id = row[0]
            error_sentence = row[1]
            sentences.append((id, error_sentence))

    return sentences

def save_data(data, save_path):
    """保存为json格式的数据"""
    pbar = tqdm(total=len(data), desc='process sampling data...')
    save_data = []
    for item in data:
        sentence = item[2]
        sampling_res = []
        choices = item[3]
        for choice in choices:
            res = choice.message.content # 采样的其中一个结果
            res_token_logprob = [] # 采样的其中一个结果的token对数概率
            # 遍历某个修正结果的每个token
            for token_msg in choice.logprobs.content:
                token_txt = token_msg.token
                # token_txt_decode = decode_token(token_txt)
                if model != 'glm-4': # glm-4的解码token直接就是汉字，这里不需要二次解码
                    token_txt_decode = decode_token(token_txt)
                else:
                    token_txt_decode = token_txt
                token_logprob = token_msg.logprob
                # token_id = token_msg.
                res_token_logprob.append({
                    'token_txt': token_txt,
                    'token_txt_decode': token_txt_decode,
                    'token_logprob': token_logprob,
                    'token_id': 1
                })
            # 存所有修正结果，以及修正结果中每个token的对数概率
            sampling_res.append({
                'response': res,
                'response_logprob': res_token_logprob
            })
        save_data.append(
            {
                'sentence': sentence,
                'sampling_responses': sampling_res
            }
        )
        pbar.update(1)

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in save_data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行


def parse_args():
    """启动前检查qyery"""
    parser = argparse.ArgumentParser(description='CGEC Inference')
    # 验证集
    parser.add_argument('--data_path', type=str, default='../data/nasgec/nasgec.exam.para', help='Test Data Path')
    parser.add_argument('--output_path', type=str, default='../data/nasgec/NASGEC_sampling_10_qwen2_7B.jsonl', help='uuid for sentences containing grammatical errors')

    return parser.parse_args()

if __name__ == '__main__':
    """
    并发推理nasgec数据集  收集采样数据
    保存的数据需要转换为score格式
    """

    args = parse_args()
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://10.10.10.157:20434/v1/",
    )

    # 和推理时设置的模型名不一定一样，反正不是glm-4模型的话就修改这个值
    # model = 'glm-4'  # 推理模型是glm-4的时候  因为glm4在输出时会在首token额外带有一个换行符，需要删除。同时其解码的token也直接是汉字，不需要再进一步解码
    model = 'qwen2.5'  # 推理模型是glm-4的时候  因为glm4在输出时会在首token额外带有一个换行符，需要删除。同时其解码的token也直接是汉字，不需要再进一步解码

    sentences = load_data(args.data_path)
    data = load_data(args.data_path)
    inference_res_list = batch_inference(data, batch_size=5)

    if model == 'glm-4':
        inference_res_list = remove_newline(inference_res_list)
    save_data(inference_res_list, args.output_path)