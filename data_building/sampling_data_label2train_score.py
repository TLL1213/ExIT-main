import argparse
import json
from tqdm import tqdm

def read_sampling_data(data_path):
    # 加载数据

    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data


def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, default='../data/fcgec/segment/FCGEC_train_sampling_10.jsonl', help='')
    parser.add_argument('--save_path', type=str, default='../data/fcgec/segment/FCGEC_train_sampling_10_train_score.jsonl', help='')
    parser.add_argument('--test_flag', type=bool, default=False, help='若为False表示现在转换的是训练数据，为True则表示测试数据，没有label信息')


    return parser.parse_args()

if __name__ == '__main__':
    """将含有label的数据转换成为训练格式，这里改成一次只输入错误句子、一个候选回答及其对应的logprobs  这是最终确定的格式"""
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    test_flag = args.test_flag



    data = read_sampling_data(data_path)

    save_data = []
    abnormal_num = 0
    pbar = tqdm(total=len(data), desc='gain score...')
    for data_idx, item in enumerate(data):
        # 获取错误句子
        sentence = item['sentence']
        sampling_responses = item['sampling_responses']

        res_list = []  # 记录所有候选，若所有候选均相同，则舍去
        res_idx_list = []  # 记录所有候选，若所有候选均相同，则舍去
        for idx, sampling_res in enumerate(sampling_responses):
            response = sampling_res['response']
            if response not in res_list:
                res_list.append(response)
                res_idx_list.append(idx)
        assert len(res_list) == len(set(res_list))
        assert len(res_idx_list) == len(set(res_idx_list))

        if len(res_idx_list) != 1: # 为1则表示所有候选都相同，这类数据没必要训练
            res_vector = []
            for idx, sampling_res in enumerate(sampling_responses):
                if idx in res_idx_list:
                    if test_flag == False:
                        score = float(sampling_res['score'])
                    response_vector = []
                    logprobs_vector = []
                    for logprob in sampling_res['response_logprob']:
                        if logprob['token_txt'] != '<|im_end|>':
                            response_vector.append(int(logprob['token_id']))
                            logprobs_vector.append(float(logprob['token_logprob']))
                    if len(response_vector) >= 1024:
                        abnormal_num += 1
                        continue
                    if test_flag == False:
                        res_vector.append({
                            'response': sampling_res['response'],
                            'response_vector': response_vector,
                            'logprobs_vector': logprobs_vector,
                            'label': score
                        })
                    else:
                        res_vector.append({
                            'response': sampling_res['response'],
                            'response_vector': response_vector,
                            'logprobs_vector': logprobs_vector,
                        })
                    save_data.append({
                        'sentence': sentence,
                        'response': res_vector
                    })
                    res_vector = []
        else:
            if test_flag == False: pass # 获得训练数据的话这不需要管，若是测试数据要保留不需要适应belief模型就行再判断的部分
            else:
                # 若所有候选回答都一样，只放入一个，这样的话最终候选也只能选这个，和计算指标时的数量就对齐了（这是为了预测结果时可以批处理进行的小技巧设置）
                response_vector = []
                logprobs_vector = []
                for logprob in sampling_res['response_logprob']:
                    if logprob['token_txt'] != '<|im_end|>':
                        response_vector.append(logprob['token_id'])
                        logprobs_vector.append(logprob['token_logprob'])
                save_data.append({
                    'sentence': sentence,
                    'response': [
                        {
                            'response': sampling_responses[0]['response'],
                            'response_vector': response_vector,  # 这里不需要进入模型，就不报错啦
                            'logprobs_vector': logprobs_vector,
                        }
                    ]
                })

        pbar.update(1)
    print(f'异常数据数量：{abnormal_num}')
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in save_data:
            json.dump(item, f, ensure_ascii=False)  # 写入 JSON 对象
            f.write("\n")  # 每个 JSON 对象占一行