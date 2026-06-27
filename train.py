import argparse
import os
import math
import json
from typing import Dict, List
from accelerate import Accelerator
import transformers
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
set_seed,
DataCollatorWithPadding,
Trainer,
TrainingArguments,
)
from transformers.trainer_utils import is_main_process

# from model.test_model5_score import BeLLMSM, BeLLMSMConfig
from model.BeliefLLMselectModel import BeLLMSM, BeLLMSMConfig # 只有一层全连接层（尽可能最轻量化） 主实验
# from model.BeliefLLMselectModel2 import BeLLMSM, BeLLMSMConfig # 多层全连接层
# from model.BeliefLLMselectModel_melt_no_log import BeLLMSM, BeLLMSMConfig # 主实验的消融
from Trainer.BeliefTrainer import BeliefTrainer
from datasets import load_dataset, Dataset
from DataCollator.BeliefDataCollator import BeliefDataCollatorWithPadding


def main(args):
    set_seed(args.seed)


    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size # 后续的embedding层的维度必须和词表长度一致，否则会索引超界，报断言错误
    special_tokens = tokenizer.special_tokens_map
    special_tokens_length = len(special_tokens)
    CAT_id = tokenizer('CAT')
    PAD_id = tokenizer('<|endoftext|>')
    CLS_id = tokenizer('CLS')

    config = BeLLMSMConfig(
        vocab_size=vocab_size + special_tokens_length,  # 这才是总词表长度
        n_layers=args.n_layers,
        Embedding_size=args.embedding_size,
        Q_size=args.Q_size,
        K_size=args.K_size,
        V_size=args.V_size,
        n_head=args.n_head,
        FF_dimension=args.FF_dimension,
        # length_tgv=args.length_tgv,
        pad_id=PAD_id['input_ids'][0],
        # max_source_length=args.max_source_length,
        max_candidate_length=args.max_candidate_length,
        drop_out=args.drop_out
    )
    model = BeLLMSM(config)

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
        label = [] # 存储真实得分
        ref_score = [] # 存储参考分数
        # 单条数据处理
        for idx, single_item in enumerate(sampling_responses):
            response_vector = []
            logprobs_vector = []
            score = []
            response_vector.extend(single_item[0]['response_vector'])
            logprobs_vector.extend(single_item[0]['logprobs_vector'])
            joint_logprob = sum(logprob for logprob in logprobs_vector)
            joint_prob_score = math.exp(joint_logprob)  # 将联合对数概率转换为联合概率
            ref_score.append([joint_prob_score])
            score.append(single_item[0]['label'])  # 回归
            input_ids = model_inputs['input_ids'][idx]
            model_inputs['input_ids'][idx] = [CLS_id['input_ids'][0]] + input_ids + [CAT_id['input_ids'][0]] + response_vector

            # if len(response_vector) <= max_candidate_length:
            #     padded_response_vector = response_vector + [PAD_id['input_ids'][0]] * (max_candidate_length - len(response_vector))
            # else:
            #     padded_response_vector = response_vector[:max_candidate_length]
            if len(logprobs_vector) <= max_candidate_length:
                # 需要在前面补齐'input_ids'和[CLS]的长度
                padded_logprobs_vector = [-0.00436] * (len(input_ids)+len(CLS_id['input_ids'])) + logprobs_vector + [-1000.0] * (max_candidate_length - len(logprobs_vector) - len(input_ids) - len(CLS_id['input_ids']))
            else:
                padded_logprobs_vector = [-0.00436] * (len(input_ids)+len(CLS_id['input_ids'])) + logprobs_vector[:max_candidate_length]
            # responses_input.append(padded_response_vector)
            logprobs_input.append(padded_logprobs_vector)
            label.append(score)
        model_inputs["labels"] = label
        model_inputs["ref_scores"] = ref_score
        model_inputs["logprobs"] = logprobs_input
        # model_inputs["responses"] = responses_input
        # keys = model_inputs.keys()
        # print("Keys in model_inputs:", list(keys))
        # print('*'*100)
        return model_inputs

    # preprocessing data

    data = load_dataset("json", data_files={"train": args.data_path})
    column_names = data['train'].column_names
    num_workers = 1 if tokenizer.__class__.__name__ == 'Qwen2Tokenizer' else os.cpu_count()
    # num_workers = 128

    accelerator = Accelerator()
    if os.path.exists(args.eval_path):
        train_ds = data["train"]
        val_ds = load_dataset("json", data_files=args.eval_path)["train"]
    elif args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=args.seed
        )
        train_ds = train_val["train"]
        val_ds = train_val["test"]
    else:
        train_ds = data["train"]
        val_ds = None

    with accelerator.main_process_first():      # first load the dataset in the main process, then load the cache in other processes
        train_data = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=True,
            remove_columns=column_names,
            # fn_kwargs={'src_max_length': args.max_source_length, 'max_candidate_length': args.max_candidate_length}
            fn_kwargs={'max_candidate_length': args.max_candidate_length}
        )
    train_data.set_format('torch')
    if val_ds is not None:
        with accelerator.main_process_first():
            val_data = val_ds.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                num_proc=num_workers,
                remove_columns=column_names,
                # fn_kwargs={'src_max_length': args.max_source_length, 'max_candidate_length': args.max_candidate_length}
                fn_kwargs={'max_candidate_length': args.max_candidate_length}
            )
        val_data.set_format('torch')
    else:
        val_data = None

    if local_rank == 0:
        print('================ dataset examples ================')
        print('max length: ', max([len(d) for d in train_data['input_ids']]))
        print(tokenizer.batch_decode(train_data['input_ids'][:2]))
        # print(tokenizer.batch_decode(train_data['labels'][:2]))
        print(train_data[0])
        print(train_data[1])
    eval_steps = 1 / args.num_train_epochs
    warmup_ratio = 1 / args.num_train_epochs
    adam_beta1, adam_beta2 = args.adam_betas
    # BeliefTrainer
    trainer = BeliefTrainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_ratio=warmup_ratio,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            # bf16=True if torch.cuda.is_bf16_supported() else False,
            # fp16=False if torch.cuda.is_bf16_supported() else True,
            fp16=True, # 注释之后，grad_norm大幅降低，但是仍然有数百
            tf32=True,
            logging_steps=args.logging_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            optim=args.optim,
            evaluation_strategy="no" if val_data is None else "steps",
            save_strategy="steps" if val_data is None else "steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=args.output_dir,
            save_total_limit=10,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=args.group_by_length,
            report_to="tensorboard",
            label_smoothing_factor=args.label_smoothing_factor,
            load_best_model_at_end=False if val_data is None else True,
            remove_unused_columns=False,  # 确保Trainer不会移除未使用的列
            max_grad_norm=10.0, # 启用梯度裁剪
            # weight_decay=0.005
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        # tokenizer=tokenizer, # 不要
        processing_class=tokenizer,
        data_collator=BeliefDataCollatorWithPadding(  # DataCollatorWithPadding/BeliefDataCollatorWithPadding 使用自定义的data_collator删除外部生成的掩码的影响
            tokenizer=tokenizer,
            padding="max_length",  # 默认为 True，表示填充到最长序列长度
            max_length=args.max_candidate_length,  # 默认为 None，表示不裁剪序列
            pad_to_multiple_of=None,  # 默认为 None，表示不填充到特定长度的倍数
            return_tensors="pt"  # 返回 PyTorch 张量
        ),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=args.patience)] if val_data is not None else [],
        alpha=args.alpha, # 自定义参数
    )



    # Training
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, 'best-model'))  # Saves the tokenizer too for easy upload

def parse_args():
    # 注意现在微调的是哪个模型
    parser = argparse.ArgumentParser(description="")

    # tokenizer/data params
    data_args = parser.add_argument_group('data', 'Data Settings')
    data_args.add_argument('--data_path', type=str, default='./data/fcgec/segment/FCGEC_train_sampling_10_train_score.jsonl', help='Training dataset path.')
    data_args.add_argument('--output_dir', type=str, default='./save_model/', help='Training dataset path.')
    data_args.add_argument('--eval_path', type=str, default='./data/fcgec/FCGEC_valid_sampling_10_train_score.jsonl', help='Eval dataset path.') # 优先级大于val_set_size，若存在路径，加载路线下的数据为验证集
    data_args.add_argument('--val_set_size', type=int, default=-1, help='') # 负数默认不切分训练集为验证集，正数表示切分多少验证集

    # tokenizer
    data_args.add_argument('--tokenizer_path', type=str, default='/root/user/cs_tcci_longlintao/LLMs/qwen2.5-14B-Instruct-fcgec-only-first-and-right', help='')
    data_args.add_argument('--num_workers', type=int, default=2, help='')

    # model hyperparams
    # model-base
    model_args = parser.add_argument_group('model', 'Model Settings')
    model_args.add_argument('--embedding_size', type=int, default=1024, help='')
    model_args.add_argument('--Q_size', type=int, default=128, help='')
    model_args.add_argument('--K_size', type=int, default=128, help='')
    model_args.add_argument('--V_size', type=int, default=128, help='')
    model_args.add_argument('--n_head', type=int, default=8, help='') # n_head*Q_size=embedding_size
    model_args.add_argument('--n_layers', type=int, default=6, help='')
    model_args.add_argument('--FF_dimension', type=int, default=4*1024, help='')
    model_args.add_argument('--drop_out', type=float, default=0.1, help='')
    model_args.add_argument('--alpha', type=float, default=0.4, help='') # 0.4、0.6、1.0、0.7、0.8


    # training hyperparams
    train_args = parser.add_argument_group('Train', 'Train Settings')
    train_args.add_argument('--seed', type=int, default=1, help='') # 2025
    train_args.add_argument('--num_train_epochs', type=int, default=60, help='')
    train_args.add_argument('--batch_size', type=int, default=128, help='') # 128 64
    train_args.add_argument('--eval_batch_size', type=int, default=32, help='') # 32 16
    train_args.add_argument('--micro_batch_size', type=int, default=128, help='') # 每个设备训练的batchsize 128 64
    # train_args.add_argument('--max_source_length', type=int, default=256, help='') # 单句最大长度，eg.原本的错误句子和每一个候选回答均最大长度为512，不足则填充到512.所以拼接以后长度为5120+512
    train_args.add_argument('--max_target_length', type=int, default=512, help='')
    train_args.add_argument('--max_candidate_length', type=int, default=300, help='') # 所有候选拼接之后的最大长度，用于填充
    train_args.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999), help='')
    train_args.add_argument('--warmup_steps', type=int, default=2000, help='')
    train_args.add_argument('--learning_rate', type=float, default=1e-5, help='')
    train_args.add_argument('--logging_steps', type=int, default=20, help='')
    train_args.add_argument('--lr_scheduler_type', type=str, default="linear", help='')
    train_args.add_argument('--optim', type=str, default="adamw_torch", help='')
    train_args.add_argument('--group_by_length', type=bool, default=False, help='')
    train_args.add_argument('--label_smoothing_factor', type=float, default=0.0, help='') # 当 label_smoothing_factor 大于 0.0 时，Trainer 会在内部自动创建一个 LabelSmoother 对象并将其赋值给 self.label_smoother。
    train_args.add_argument('--patience', type=int, default=10, help='') # 5、10

    return parser.parse_args()

if __name__ == '__main__':
    """
    在train2的基础上，修改预处理函数和loss计算，引入LLM对不同句子的整体句子打分，来共同计算损失
    修改了数据处理函数，首字符引入了CLS
    """
    args = parse_args()
    print(args)
    main(args)