import argparse
import os
from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
import json
# 确保 BeLLMSM 和 BeLLMSMConfig 类在预测脚本中可用
# 假设 model/test_model2.py 包含这两个类的定义
# from model.test_model5_score import BeLLMSM
from model.BeliefLLMselectModel import BeLLMSM
from tqdm import tqdm
from DataCollator.BeliefDataCollator import BeliefDataCollatorWithPadding

def predict(args):


    # 加载分词器
    model_save_directory = os.path.join(args.model_path, 'best-model')
    # model_save_directory = os.path.join(args.model_path, 'checkpoint-2583')

    print(f"Loading tokenizer from: {model_save_directory}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("请确保分词器已正确保存。")
        return  # 如果分词器加载失败，终止程序

    # 获取 PAD_id
    PAD_id = tokenizer('<|endoftext|>')
    CAT_id = tokenizer('CAT')
    CLS_id = tokenizer('CLS')

    # 3. 加载模型
    print(f"Loading BeLLMSM model from: {model_save_directory}...")
    try:
        # 直接使用你的自定义模型类 BeLLMSM 的 from_pretrained 方法
        model = BeLLMSM.from_pretrained(model_save_directory)
        print("BeLLMSM model loaded successfully!")
    except Exception as e:
        print(f"Error loading BeLLMSM model: {e}")
        print(
            "请确保 BeLLMSMConfig 和 BeLLMSM 类已正确定义，并且与保存模型的配置匹配。"
        )
        return  # 如果加载失败，终止程序

    # 将模型设置为评估模式并移动到指定设备
    model.eval()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id != -1 else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    # 4. 数据预处理函数
    def preprocess_function(batch: Dict[str, List], max_candidate_length):
        """
        一次只输入错误句子、一条候选及其相应的logprob
        :param batch:
        :param src_max_length: 输入句子的最大长度
        :param max_candidate_length: 错误句子和一个候选拼接后的最大长度
        :return:
        """
        sentences = batch['sentence']
        model_inputs = tokenizer(sentences,
                                 # max_length=src_max_length,
                                 padding=False,
                                 truncation=False,
                                 return_token_type_ids=False)
        del model_inputs['attention_mask']
        sampling_responses = batch['response']
        # responses_input = []
        logprobs_input = []
        label = []
        # 单条数据处理
        for idx, single_item in enumerate(sampling_responses):
            response_vector = []
            logprobs_vector = []
            response_vector.extend(single_item[0]['response_vector'])
            logprobs_vector.extend(single_item[0]['logprobs_vector'])
            input_ids = model_inputs['input_ids'][idx]
            model_inputs['input_ids'][idx] = [CLS_id['input_ids'][0]] + input_ids + [CAT_id['input_ids'][0]] + response_vector

            # if len(response_vector) <= max_candidate_length:
            #     padded_response_vector = response_vector + [PAD_id['input_ids'][0]] * (
            #                 max_candidate_length - len(response_vector))
            # else:
            #     padded_response_vector = response_vector[:max_candidate_length]
            if len(logprobs_vector) <= max_candidate_length:
                # 需要在前面补齐'input_ids'的长度
                padded_logprobs_vector = [-0.00436] * (len(input_ids)+len(CLS_id['input_ids'])) + logprobs_vector + [-1000.0] * (
                            max_candidate_length - len(logprobs_vector) - len(input_ids) - len(CLS_id['input_ids']))
            else:
                padded_logprobs_vector = [-0.00436] * (len(input_ids)+len(CLS_id['input_ids'])) + logprobs_vector[:max_candidate_length]
            # responses_input.append(padded_response_vector)
            logprobs_input.append(padded_logprobs_vector)
        model_inputs["logprobs"] = logprobs_input
        return model_inputs

    # 5. 加载和预处理输入数据
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print(f"Loading input data from: {args.input_file}...")
    # 假设输入文件也是jsonl格式，包含 'sentence' 字段
    input_data = Dataset.from_json(args.input_file)
    # input_data = input_data.select(range(10)) # 做后续逻辑验证用，正式预测进行注释
    # 移除原始列，只保留模型需要的列
    processed_input_data = input_data.map(
        preprocess_function,
        batched=True,
        remove_columns=input_data.column_names,  # 移除原始列，因为我们只关心处理后的input_ids
        fn_kwargs={'max_candidate_length': args.max_candidate_length}
    )
    processed_input_data.set_format('torch')  # 设置为 PyTorch 张量格式

    print("Input data preprocessed successfully!")
    if len(processed_input_data) > 0:
        print('================ Input data example ================')
        print(f"Original sentence: {input_data[0]['sentence']}")
        print(f"Tokenized input_ids: {processed_input_data[0]['input_ids']}")
        print(f"Decoded input: {tokenizer.decode(processed_input_data[0]['input_ids'])}")
    else:
        print("Warning: Input data is empty after preprocessing.")

    # DataCollator，这里主要用于填充
    data_collator = BeliefDataCollatorWithPadding( # BeliefDataCollatorWithPadding/DataCollatorWithPadding
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_candidate_length,
        return_tensors="pt"
    )

    # 创建 DataLoader
    # 在预测时，batch_size 可以是 eval_batch_size
    prediction_dataloader = DataLoader(
        processed_input_data,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        shuffle=False  # 预测时不需要打乱
    )

    # 6. 进行预测
    print("Starting prediction...")
    predictions = []
    pbar = tqdm(total=len(input_data))
    # 如果你的模型输出包含多个部分，例如分数、响应等，你可能需要更复杂的逻辑来收集它们
    with torch.no_grad():  # 预测时不需要计算梯度
        for batch_idx, batch in enumerate(prediction_dataloader):
            # 将batch中的数据移动到正确的设备
            input_ids = batch['input_ids'].to(device)
            logprobs = batch['logprobs'].to(device)
            # responses = batch['responses'].to(device)

            # 调用模型进行前向传播
            # 这里的输出结构取决于你的 BeLLMSM 模型的 forward 方法
            # 假设 forward 方法返回一个包含 'logits' 或直接是预测结果的字典或元组
            # 你需要根据 BeLLMSM 的实际输出调整这里
            model_output = model(input_ids=input_ids, logprobs=logprobs)
            # model_output = torch.sigmoid(model_output)
            batch_predictions = model_output.cpu().numpy()
            predictions.extend(batch_predictions)

            pbar.update(args.eval_batch_size)

    print("Prediction complete!")




    # 保存预测结果
    if args.output_file:
        print(f"Saving predictions to {args.output_file}...")
        sentence = input_data['sentence']
        response = input_data['response']
        save_data = []
        for i, pred in enumerate(predictions):
            temp = {
                'sentence': sentence[i],
                'res': response[i][0]['response'],
                'score': float(pred[0]) # 适应json格式原本的数据类型
            }
            save_data.append(temp)
        # 进行选择
        end_data = {}
        end_idx = 0
        signal_sentence = [] # 用于判断不同的数据是否属于同一错误句子
        select_res = [] # 用于判断同一错误句子下，选择哪个候选
        for item in save_data:
            if len(signal_sentence) == 0: # 确定当前进行判断的错误句子
                signal_sentence.append(item['sentence'])
                select_res.append(item)
            else:
                if item['sentence'] in signal_sentence:
                    select_res.append(item)
                else:
                    end_data[end_idx] = select_res
                    end_idx += 1
                    # 当前这一个错误句子和上一个不同 这里初始化时得加入
                    signal_sentence = [item['sentence']]
                    select_res = [item]

        end_data[end_idx] = select_res # 加入最后一个元素
        # 遍历最终答案，选择合适的输出
        out_data = {}
        for key, value in end_data.items():
            selected_response = ''
            max_score = -9999999
            for item in value:
                if item['score'] > max_score:
                    max_score = item['score']
                    selected_response = item['res']
            out_data[key] = {
                'sentence': value[0]['sentence'],
                'res': selected_response,
                'score': max_score
            }
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=4)
        f.close()
        print("Predictions saved successfully!")
    else:
        print("Output file not specified, printing first few predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Input: {input_data[i]['sentence']} -> Prediction: {predictions[i]}")


def parse_args_predict():
    parser = argparse.ArgumentParser(description="BeLLMSM Model Prediction Script")

    # 模型和数据路径
    parser.add_argument('--model_path', type=str, default='./save_model', help='Path to the directory containing the saved BeLLMSM model and tokenizer.')

    # fcgec-test
    parser.add_argument('--input_file', type=str, default='./data/fcgec/FCGEC_test_only_error_sampling_10_train_score.jsonl', help='Path to the input JSONL file containing sentences for prediction.')
    parser.add_argument('--output_file', type=str, default='./data/fcgec/predict/predict.json', help='Path to save the prediction results. If not specified, prints to console.')

    # nasgec-exam
    # parser.add_argument('--input_file', type=str, default='./data/nasgec/NASGEC_exam_sampling_10_train_score.jsonl', help='qwen2.5-14B')
    # parser.add_argument('--output_file', type=str, default='./data/nasgec/predict/predict.json', help='Path to save the prediction results. If not specified, prints to console.')

    # nacgec
    # parser.add_argument('--input_file', type=str, default='./data/nacgec/NACGEC_sampling_10_train_score.jsonl', help='Path to the input JSONL file containing sentences for prediction.')
    # parser.add_argument('--output_file', type=str, default='./data/nacgec/predict/predict.json', help='Path to save the prediction results. If not specified, prints to console.')

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for prediction. Set to -1 for CPU.')

    # 预测超参数
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for prediction.')
    # parser.add_argument('--max_source_length', type=int, default=512, help='Maximum input sequence length.')
    parser.add_argument('--max_candidate_length', type=int, default=300, help='')  # 所有候选拼接之后的最大长度，用于填充

    return parser.parse_args()


if __name__ == '__main__':
    """增加了CLS"""
    args = parse_args_predict()
    predict(args)