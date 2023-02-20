# -*- coding: utf-8 -*-

import pandas as pd
import jieba
import re


"""
函数说明：简单分词
Parameters:
     filename:数据文件
Returns:
     list_word_split：分词后的数据集列表
     category_labels: 文本标签列表
"""

jieba.load_userdict("/home/luojiawei/mimic3_R_work/THUOCL_medical.txt")

def cleantext(raw_text):
    list_row_data = list(jieba.cut(raw_text))  # 对单个漏洞进行分词
    list_row_data1 = []
    for x in list_row_data:
        x = re.sub("[^a-zA-Z]+", "", x)
        x = re.sub("( )+", "", x)
        list_row_data1.append(x)
    list_row_data = [x.lower() for x in list_row_data1 if x != ' ' and x.lower() not in stop_words]  # 去除列表中的空格字符
    list_row_data = " ".join(list_row_data)
    return list_row_data


def cleantext1(raw_text):
    pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
    x = pattern.sub('', raw_text)
    x = re.sub("[0-9]+", "", x)
    x = re.sub("( )+", " ", x)
    return x


if __name__ == '__main__':

    global stop_words

    filename = "/home/luojiawei/mimic3_R_work/stopwords.txt"
    with open(filename, 'r') as fr:
        stop_words = list(fr.read().split('\n'))  # 将停用词读取到列表里

    filename = "/home/luojiawei/mimic3_R_work/noteevents1.csv"
    read_data = pd.read_csv(filename, header=0)
    raw_text = read_data.iloc[:, 10].tolist()
    clean_text = map(cleantext, raw_text)
    read_data['text1'] = list(clean_text)
    read_data.to_csv("/home/luojiawei/mimic3_R_work/noteevents_已经分词2.csv", header=True)
