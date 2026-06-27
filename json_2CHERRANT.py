import json
import re

def remove_punctuation(text):
    all_punctuation = r"""！？｡。，,.;:《》'$%^&*()!@#~`?/\|<>"“”：；、%&……！＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    # 使用正则表达式去除标点符号
    return re.sub(f"[{re.escape(all_punctuation)}]", "", text)

if __name__ == '__main__':
    # fcgec-test
    json_path = './data/fcgec/predict/predict.json'
    save_path = './data/fcgec/predict/predict.hyp.para'
    mv_punctuation = True # 是否移除标点符号，仅仅只有FCGEC数据集需要


    # nasgec-exam
    # json_path = './data/nasgec/predict/predict.json'
    # save_path = 'data/nasgec/predict/nasgec_exam_predict.hyp.para'
    # mv_punctuation = False

    # nacgec
    # json_path = './data/nacgec/predict/predict.json'
    # save_path = 'data/nacgec/predict/nacgec.test.predict.hyp.para'
    # mv_punctuation = False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()

    save_data = []
    for k, v in data.items():
        sentence = v['sentence'].strip()
        # correction = v['res'].strip()
        if '修正形式：' in v['res']:
            correction = v['res'].replace('修正形式：', '').strip()
        else:
            correction = v['res'].strip()
        save_data.append((sentence, correction))

    # # 保存文件
    with open(save_path, 'w', encoding='utf-8') as f:
        idx = 1
        for item in save_data:
            if mv_punctuation == True:
                err_s = remove_punctuation(item[0])
                res = remove_punctuation(item[1])
            else:
                err_s = item[0]
                res = item[1]
            f.write(f'{idx}\t{err_s}\t{res}\n')
            idx += 1