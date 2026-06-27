import argparse
import json
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
# import jieba
import math
import re

few_shot_prompt = """一些参考示例如下：
1.
原始句子：改革开放搞活了经济，农贸市场的货物琳琅满目，除各种应时的新鲜蔬菜外，还有肉类、水产品、鱼、虾、甲鱼、牛蛙及各种调味品。
修正形式：改革开放搞活了经济，农贸市场的货物琳琅满目，除各种应时的新鲜蔬菜外，还有肉类、鱼、虾、甲鱼、牛蛙等水产品及各种调味品。
2.
原始句子：2024年春季招聘会上，国家图书馆与各分馆共推出200多个职位，主要招聘本科生和研究生，为图书馆的数字化进程培养、储备书籍。
修正形式：2024年春季招聘会上，国家图书馆与各分馆共推出200多个职位，主要招聘本科生和研究生，为图书馆的数字化进程培养、储备人才。
3.
原始句子：李教授主编的《现代汉语词典》是大学出版社发行首次权威工具书，全书由资深编辑团队细致校对。
修正形式：李教授主编的《现代汉语词典》是大学出版社首次发行权威工具书，全书由资深编辑团队细致校对。
4.
原始句子：人生价值和意义，其实并不在于别人对自己如何膜拜、崇敬、羡慕，而在于自己对社会，对历史的进步和发展作出何种贡献。
修正形式：人生价值和意义，其实并不在于别人对自己如何羡慕、崇敬、膜拜，而在于自己对社会，对历史的进步和发展作出何种贡献。
5.
原始句子：在会议上，教育局局长李明提出，正是学生们的勤奋与努力，老师们的教导与奉献，汇聚成推动教育进步的动力，成为鼓动学校发展的灯塔。
修正形式：在会议上，教育局局长李明提出，正是学生们的勤奋与努力，老师们的教导与奉献，汇聚成推动教育进步的动力，成为引导学校发展的灯塔。
6.
原始句子：随着20多年来中国经济的迅猛发展，以及中国与全球经济的不断融合，解决“三农”问题随之变得更加紧迫。
修正形式：20多年来中国经济的迅猛发展，以及中国与全球经济的不断融合，解决“三农”问题随之变得更加紧迫。
7.
原始句子：我们的报纸、杂志、电视和一切出版物，更有责任作出表率，杜绝用字不规范现象，增强使用文字的规范意识。
修正形式：我们的电视和一切出版物如报纸、杂志等，更有责任作出表率，杜绝用字不规范现象，增强使用文字的规范意识。
8.
原始句子：少吃主食、水果和含糖食物的低碳饮食可以在数周内较快降低体重，是因为水分、肌肉和脂肪一起减少了。
修正形式：少吃主食、水果等含糖食物的低碳饮食可以在数周内较快降低体重，是因为水分、肌肉和脂肪一起减少了。
9.
原始句子：3月15日，“全球最大电动汽车充电网络启动仪式暨智能充电技术论坛”在北京国际会展中心开幕。此次盛会被新闻网站、电视台、广播电台、媒体等广泛报道。
修正形式：3月15日，“全球最大电动汽车充电网络启动仪式暨智能充电技术论坛”在北京国际会展中心开幕。此次盛会被新闻网站、电视台、广播电台等广泛报道。
10.
原始句子：截至2023年底，该公司在国际市场上共推出新产品50多款，对提升公司形象、公司文化“走向世界”发挥了积极作用。
修正形式：截至2023年底，该公司在国际市场上共推出新产品50多款，对提升公司形象、促进公司文化“走向世界”发挥了积极作用。
11.
原始句子：当下许多“零门槛”的选秀节目，让拥有才华和梦想的普通人都可以展示自我，也让人深信：平凡人成就自我的关键在于是否相信梦想，相信奇迹。
修正形式：当下许多“零门槛”的选秀节目，让拥有才华和梦想的普通人都可以展示自我，也让人深信：平凡人成就自我的关键在于相信梦想，相信奇迹。
12.
原始句子：读了这本书关于未来科技趋势的分析，激发了我们的兴趣，丰富了我们的知识，提高了我们对未来科技发展的洞察力。
修正形式：这本书关于未来科技趋势的分析，激发了我们的兴趣，丰富了我们的知识，提高了我们对未来科技发展的洞察力。
13.
原始句子：留学生们如果能为自己找准清晰的定位，为自己而留学，那么社会上“留学归国后找不到工作”的抱怨，自然就会减少很多。
修正形式：如果留学生们能为自己找准清晰的定位，为自己而留学，那么社会上“留学归国后找不到工作”的抱怨，自然就会减少很多。
14.
原始句子：《中国诗词大会》节目致力于弘扬中国传统文化，自开播以来深受观众喜爱的原因是其新颖的内容和多样的形式造成的。
修正形式：《中国诗词大会》节目致力于弘扬中国传统文化，自开播以来深受观众喜爱的原因是其新颖的内容和多样的形式。
15.
原始句子：“和平使命——2009”联合演习是中俄两国继“和平使命——2005”、“和平使命——2007”演习后，在上海合作组织框架内举行的又一次大规模防恐演习。演习分战役准备、战略磋商和战役实施三个阶段。
修正形式：“和平使命——2009”联合演习是中俄两国继“和平使命——2005”、“和平使命——2007”演习后，在上海合作组织框架内举行的又一次大规模防恐演习。演习分战略磋商、战役准备和战役实施三个阶段。
16.
原始句子：北京故宫博物院最新展出的古代文物，吸引了众多游客，展览布局、展品极具特色，展览中的瓷器也成了摄影爱好者关注的焦点。
修正形式：北京故宫博物院最新展出的古代文物，吸引了众多游客，展览布局巧妙、展品极具特色，展览中的瓷器也成了摄影爱好者关注的焦点。
17.
原始句子：去年，华为宣布成功研发5G芯片，正式结束通信设备巨头每年投入10亿美元进口芯片。
修正形式：去年，华为宣布成功研发5G芯片，正式结束通信设备巨头每年投入10亿美元进口芯片的历史。
18.
原始句子：这次偷渡人数竟多达15人以上，属于有组织、有预谋的重大偷渡案件，因此，被警方定名为“1?06特大偷渡案”。
修正形式：这次偷渡人数竟多15人以上，属于有组织、有预谋的重大偷渡案件，因此，被警方定名为“1?06特大偷渡案”。
19.
原始句子：中国国防科工委官员介绍说，中巴地球资源卫星的成功发射、应用及运行，使世界上许多国家对中国的地球卫星产生了浓厚兴趣。
修正形式：中国国防科工委官员介绍说，中巴地球资源卫星的成功发射、运行及应用，使世界上许多国家对中国的地球卫星产生了浓厚兴趣。
20.
原始句子：中国高铁不仅运营规模大，而且还具有系统技术全面、造价低、建设速度快，成为“中国速度”“中国制造”的新名片。
修正形式：中国高铁不仅运营规模大，而且还具有系统技术全面、造价低、建设速度快的特点，成为“中国速度”“中国制造”的新名片。
21.
原始句子：“自从知道我被选中去进行首次飞行时，我的心里就无时无刻不忘为祖国振作精神，争得荣誉。”杨利伟谈到。
修正形式：“自从知道我被选中去进行首次飞行时，我的心里就每时每刻都为祖国振作精神，争得荣誉。”杨利伟谈到。
22.
原始句子：不仅文明细节是思想上的自觉、精神上的自制，也是一种行为上的规范。
修正形式：文明细节不仅是思想上的自觉、精神上的自制，也是一种行为上的规范。
23.
原始句子：经过一番激烈的讨价还价，对方终于作出了让步，最终价格定在9000元，比原来的2万多元少了一倍还多。
修正形式：经过一番激烈的讨价还价，对方终于作出了让步，最终价格定在9000元，比原来的2万多元少了一半还多。

你仅仅需要输出“修正形式：”之后的最终答案，不要输出其他内容。输入句子如下：
"""

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

def query(input_id, messages):
    """访问openai接口的函数"""
    # system设置风格
    error_sentence = messages['messages'][1]['content']

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
            {"role": "user", "content": f"{error_sentence}"},
            # {"role": "user", "content": f"{few_shot_prompt}{error_sentence}"},
        ],
        # messages=messages['messages'], # 不使用few_shot
        model=model_name,
        n=10,
        logprobs=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # 不是qwen3的话注销
    )

    output = chat_completion.choices

    return input_id, error_sentence, output

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
                executor.submit(query, input_id=idx, messages=tmp) for idx, tmp in enumerate(batch)
            ]
            for future in concurrent.futures.as_completed(futures):
                # future.result()
                input_id, error_sentence, res = future.result()  # 获取返回值
                batch_res.append((input_id, error_sentence, res))  # 将返回值存储到列表中
        # 因为并发的缘故，这里的input_id不是递增的，所以重新按照input_id排序
        batch_res = sorted(batch_res, key=lambda x: x[0])
        inference_res_list += batch_res
        if i + batch_size < len(data):
            pbar.update(batch_size)
        else:
            pbar.update(len(data) - i)
    return inference_res_list




def save_data(data, save_path):
    """保存为json格式的数据"""
    pbar = tqdm(total=len(data), desc='process sampling data...')
    save_data = []
    for item in data:
        sentence = item[1]
        sampling_res = []
        choices = item[2]
        for choice in choices:
            res = choice.message.content # 采样的其中一个结果
            res_token_logprob = [] # 采样的其中一个结果的token对数概率
            # 遍历某个修正结果的每个token
            for token_msg in choice.logprobs.content:
                token_txt = token_msg.token
                token_txt_decode = decode_token(token_txt)
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


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            messages = []
            messages.append(item['messages'][0])
            messages.append(item['messages'][1])
            data.append({
                'messages': messages
            })

    return data

def parse_args():
    # 需要注意是直接纠错还是基于few-shot纠错  跑之前去query函数检查一下，也可能是qwen3  反正去检查query  以防设置错
    parser = argparse.ArgumentParser(description='CGEC Inference')

    parser.add_argument('--data_path', type=str, default='../data/fcgec/segment/FCGEC_train_section3.jsonl', help='')
    parser.add_argument('--output_path', type=str, default='../data/fcgec/segment/FCGEC_train_section3_sampling_10.jsonl', help='uuid for sentences containing grammatical errors')

    # parser.add_argument('--data_path', type=str, default='../data/fcgec/segment/FCGEC_train_section0.jsonl', help='Test Data Path')
    # parser.add_argument('--output_path', type=str, default='../data/fcgec/segment/FCGEC_train_section0_sampling_10_with_few_shot.jsonl', help='uuid for sentences containing grammatical errors')
    parser.add_argument('--model_name', type=str, default='lora1', help='') # qwen2.5

    return parser.parse_args()

if __name__ == '__main__':
    """
    并发推理fcgec数据的训练集  收集采样数据进行训练
    这里采样的是经过K-折之后得到的训练数据
    """

    args = parse_args()
    model_name = args.model_name
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://10.10.10.157:20434/v1/",
    )

    data = load_data(args.data_path)
    inference_res_list = batch_inference(data, batch_size=5)
    save_data(inference_res_list, args.output_path)


