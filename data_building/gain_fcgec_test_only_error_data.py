import json



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

    save_data = {}
    for item in test_error_keys:
        save_data[item] = {
            "sentence": fcgec_test_data[item]["sentence"],
            "version": "FCGEC EMNLP 2022"
        }

    return save_data

if __name__ == '__main__':
    """因为FCGEC的公布的测试文件（带去符号label）中只测试含有错误句子的部分，所以这里将不含有错误句子的部分进行剔除"""

    data_path = '../data/fcgec/FCGEC_test.json'
    test_error_key_path = '../data/fcgec/FCGEC_test_error_keys.txt'
    out_path = '../data/fcgec/FCGEC_test_only_error.json'

    fcgec_test_data_only_error = load_test_data_only_error(test_error_key_path, data_path)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fcgec_test_data_only_error, f, ensure_ascii=False, indent=4)
