## FCGEC数据额外说明
路径`data/fcgec/right_FCGEC`下的csv文件为原作者代码转换的seq2seq格式数据。
其中`ref.hyp.para`是从原作者提供的开源评测平台上得到，可以用于本地评测。可以搭配`data_building/gain_fcgec_test_only_error_data`获取。
本项目已将处理后的FCGEC测试集`FCGEC_test_only_error.json`放入文件夹`data/fcgec`。

原始数据集格式（.json）请从原作者处下载。