import argparse
import json
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import jieba
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

input_prompt = '人们在使用语言进行交流沟通的时候，喜欢使用一些约定俗称的形式，即便一句话有多种表达方式，但是人们倾向于使用一种最没有歧义，且格式易于分析的形式，下面请你将输入的句子转换为满足人们偏好的形式，输入句子如下：\n'

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



def edit_distance(sentence1, sentence2):
    words1 = jieba.lcut(sentence1)
    words2 = jieba.lcut(sentence2)
    # words1 = sentence1.split()
    # words2 = sentence2.split()

    # 初始化编辑距离矩阵
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]

    # 填充矩阵
    for i in range(1, len(words1) + 1):
        dp[i][0] = i
    for j in range(1, len(words2) + 1):
        dp[0][j] = j

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # 删除
                               dp[i][j - 1] + 1,  # 插入
                               dp[i - 1][j - 1] + 1)  # 替换

    return dp[-1][-1]


def load_test_data_only_error(test_error_keys_path, test_data_path):
    """仅仅加载测试集中错误的句子（按顺序）"""
    # 加载测试集中含有语法错误句子的key
    test_error_keys = []
    with open(test_error_keys_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_error_keys.append(line.strip())

    # 加载fcgec测试集
    with open(test_data_path, 'r', encoding='utf-8') as f:
        fcgec_test_data = json.load(f)

    test_error_sentences = []
    for item in test_error_keys:
        test_error_sentences.append((item, fcgec_test_data[item]['sentence']))

    return test_error_sentences

# def calculate_confidence(responses):
#     """计算整个句子的置信度，传入了多少个采样句子，返回对应数量的置信度"""
#     confidence_list=[]
#     for choice in responses:
#         logprobs = choice.logprobs.content
#         # joint_logprob = sum(log.top_logprobs[0].logprob for log in logprobs)# 计算整个句子的联合对数概率
#         joint_logprob = sum(log.logprob for log in logprobs)  # 计算整个句子的联合对数概率
#         joint_prob = math.exp(joint_logprob)# 将联合对数概率转换为联合概率
#         joint_confidence = joint_prob * 100# 将联合概率转换为百分比形式的联合置信度
#         confidence_list.append(joint_confidence)
#     return confidence_list

def calculate_confidence(responses):
    """计算整个句子的置信度，传入了多少个采样句子，返回对应数量的置信度
    增加鲁棒性，在使用few-shot的过程中，有的回答不满足条件，会模仿示例额外在前面输出‘修正形式：’，
    为了避免该内容造成的影响，这里删除该额外字符串造成的影响"""
    confidence_list=[]
    for choice in responses:
        logprobs = choice.logprobs.content
        # joint_logprob = sum(log.top_logprobs[0].logprob for log in logprobs)# 计算整个句子的联合对数概率
        if '修正形式：' not in choice.message.content:
            joint_logprob = sum(log.logprob for log in logprobs)  # 计算整个句子的联合对数概率
        else:
            joint_logprob = 0
            extra_token_num = 0 # 这里只消除额外输出的“修正形式：”的影响，对于后面修正句子中出现的这几个token则需要继续考虑其概率
            extra_token_list = ['修正', '形式', '：']
            for log in logprobs:
                if extra_token_num >= len(extra_token_list):
                    joint_logprob += log.logprob
                    continue
                if log.token not in ['修正', '形式', '：']:
                    joint_logprob += log.logprob
                else:
                    extra_token_num += 1

        joint_prob = math.exp(joint_logprob)# 将联合对数概率转换为联合概率
        joint_confidence = joint_prob * 100# 将联合概率转换为百分比形式的联合置信度
        confidence_list.append(joint_confidence)
    return confidence_list

def query2(input_id, uuid, error_sentence):
    """访问openai接口的函数 这里同时获得不同prompt的内容"""

    def gain_confidence_res(chat_completion):
        # 选择其中编辑距离最小的，若多个编辑距离均最小，则选择置信度最大的选项
        responses_text = []
        responses = []
        for choice in chat_completion.choices:
            # 去重
            if choice.message.content not in responses_text:
                responses_text.append(choice.message.content)
                responses.append(choice)
        if len(responses_text) > 1:
            # *****************************************************************************************************************
            # 只选择置信度最大的
            confidence_list = calculate_confidence(responses)
            max_confidence = max(confidence_list)
            min_index = confidence_list.index(max_confidence)
            output = responses_text[min_index]
            # *****************************************************************************************************************
        else:
            output = responses_text[0]
            confidence_list = calculate_confidence(responses)
            max_confidence = confidence_list[0]
        output = output.strip()
        return output, max_confidence

    # system设置风格
    chat_completion_sero_shot = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
            {"role": "user", "content": f"{error_sentence}"},
        ],
        model="qwen2.5",
        # model="glm-4-9b-chat",
        # top_p=0.7,
        # temperature=0.65,
        n=10,
        logprobs=True,
        # top_logprobs=1
    )
    chat_completion_few_shot = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
            {"role": "user", "content": f"{few_shot_prompt}{error_sentence}"},
        ],
        model="qwen2.5",
        # model="glm-4-9b-chat",
        # top_p=0.7,
        # temperature=0.65,
        n=10,
        logprobs=True,
        # top_logprobs=1
    )

    out_sero_shot, max_confidence_sero_shot = gain_confidence_res(chat_completion_sero_shot)
    out_few_shot, max_confidence_few_shot = gain_confidence_res(chat_completion_few_shot)
    if out_sero_shot == out_few_shot:
        output = out_sero_shot
    else:
        if max_confidence_sero_shot > max_confidence_few_shot:
            output = out_sero_shot
        else:
            output = out_few_shot

    return input_id, uuid, error_sentence, output

def query(input_id, uuid, error_sentence):
    """访问openai接口的函数"""
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": f'{input_prompt}{error_sentence}',
    #         }
    #     ],
    #     model="qwen2.5",
    #     # top_p=0.7,
    #     # temperature=0.65,
    #     n=10,
    #     # presence_penalty=0.5,
    #     # frequency_penalty=0.5,
    #     logprobs=True,
    #     # top_logprobs=1
    # )

    # system设置风格
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个有用的中文文本修正助手，你可以纠正中文句子中存在的错误。"},
            {"role": "user", "content": f"{error_sentence}"},
            # {"role": "user", "content": f"{few_shot_prompt}{error_sentence}"},
        ],
        model="qwen2.5",
        # model="glm-4-9b-chat",
        # top_p=0.7,
        # temperature=0.65,
        # temperature=0,
        # n=10,
        # presence_penalty=0.5,
        # frequency_penalty=0.5,
        # logprobs=True,
        # top_logprobs=1
    )

    # # 选择其中置信度最大的选项
    # responses_text = []
    # responses = []
    # for choice in chat_completion.choices:
    #     # 去重
    #     if choice.message.content not in responses_text:
    #         responses_text.append(choice.message.content)
    #         responses.append(choice)
    # if len(responses_text) > 1:
    #     # # *****************************************************************************************************************
    #     # # 先计算编辑距离，若多个编辑距离均为最小，则选择其中置信度最高的
    #     # edit_distance_list = [edit_distance(error_sentence, res) for res in responses_text]  # 计算错误句子和模型修正直接的词级别编辑距离，传入了多少个采样句子，返回对应数量的编辑距离计算
    #     # min_distance = min(edit_distance_list)
    #     # min_edit_distance_index_list = [index for index, value in enumerate(edit_distance_list) if value == min_distance]
    #     #
    #     # # 只用最小编辑判断
    #     # # output = responses_text[min_edit_distance_index_list[0]]
    #     #
    #     # # 对于有多个候选均是最小编辑距离，则用与错误句子最相似的结果
    #     # max_cos = -1.0  # 最大余弦相似度
    #     # # min_cos = 9999  # 最小余弦相似度
    #     # if len(min_edit_distance_index_list) == 1:
    #     #     output = responses_text[min_edit_distance_index_list[0]]
    #     # else:
    #     #     for res in responses_text:
    #     #         tfidf_matrix = vectorizer.fit_transform([error_sentence, res])
    #     #         # 计算余弦相似度
    #     #         cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    #     #         # 最好
    #     #         if max_cos < cos_sim[0][0]:
    #     #             max_cos = cos_sim[0][0]
    #     #             output = res
    #     #         # # 最坏
    #     #         # if min_cos > cos_sim[0][0]:
    #     #         #     min_cos = cos_sim[0][0]
    #     #         #     output = res
    #     # # *****************************************************************************************************************
    #
    #     # 对于有多个候选均是最小编辑距离，则用置信度判断
    #     # if len(min_edit_distance_index_list) == 1:
    #     #     output = responses_text[min_edit_distance_index_list[0]]
    #     # else:
    #     #     confidence_list = calculate_confidence(responses)
    #     #     confidence_list_sub = [confidence_list[idx] for idx in min_edit_distance_index_list]
    #     #     max_confidence = max(confidence_list_sub)
    #     #     max_index = confidence_list.index(max_confidence)
    #     #     output = responses_text[max_index]
    #     # *****************************************************************************************************************
    #
    #     # *****************************************************************************************************************
    #     # 只选择置信度最大的
    #     confidence_list = calculate_confidence(responses)
    #     max_confidence = max(confidence_list)
    #     min_index = confidence_list.index(max_confidence)
    #     output = responses_text[min_index]
    #     # *****************************************************************************************************************
    # else:
    #     output = responses_text[0]
    # 不做任何操作
    output = chat_completion.choices[0].message.content
    output = output.strip()
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
                # executor.submit(query2, input_id=idx, uuid=tmp[0], error_sentence=tmp[1]) for idx, tmp in enumerate(batch)
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

def save_response(inference_res_list, save_path):
    """将推理结果保存"""
    save_data = {}
    pbar = tqdm(total=len(inference_res_list), desc='save inference response.')
    for item in inference_res_list:
        uuid = item[1]
        error_sentence = item[2]
        correction = item[3]
        save_data[uuid] = {
            "sentence": error_sentence,
            "res": correction
        }
        pbar.update(1)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='CGEC Inference')
    parser.add_argument('--test_data_path', type=str, default='./data/fcgec/FCGEC_test.json', help='Test Data Path')
    parser.add_argument('--test_error_key_path', type=str, default='./data/fcgec/FCGEC_test_error_keys.txt', help='uuid for sentences containing grammatical errors')
    parser.add_argument('--output_path', type=str, default='./data/fcgec/predict/predict.json', help='uuid for sentences containing grammatical errors')
    parser.add_argument('--top_p', type=float, default=0.7, help='')
    parser.add_argument('--temperature', type=float, default=0.65, help='')

    return parser.parse_args()

if __name__ == '__main__':
    """
    并发推理fcgec数据的测试集  句子级情况下去选择模型输出的多个答案
    核心是一次查询采样多个回答，去确定多个回答谁更合适。
    """

    args = parse_args()
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://10.10.10.168:20151/v1/",
    )

    # 创建TF-IDF模型
    vectorizer = TfidfVectorizer()

    test_error_sentences = load_test_data_only_error(args.test_error_key_path, args.test_data_path)
    inference_res_list = batch_inference(test_error_sentences, batch_size=5)
    save_response(inference_res_list, args.output_path)
    print('a')
