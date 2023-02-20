# -*- coding: utf-8 -*-
 
import pandas as pd
import jieba
from nltk.stem import WordNetLemmatizer
import re
 
"""
函数说明：简单分词
Parameters:
     filename:数据文件
Returns:
     list_word_split：分词后的数据集列表
     category_labels: 文本标签列表
"""
def cleantext(raw_text):
    list_row_data = list(jieba.cut(raw_text)) # 对单个漏洞进行分词
    list_row_data= [re.sub("[^a-zA-Z0-9]+","",x) for x in list_row_data]
    list_row_data= [x.lower() for x in list_row_data if x!=' ' and x.lower() not in stop_words] #去除列表中的空格字符
    list_row_data = " ".join(list_row_data)
    return list_row_data
 
 
if __name__=='__main__':
    import sys
    global stop_words
    
    filename = "./stopwords.txt"
    with open(filename,'r') as fr:
        stop_words=list(fr.read().split('\n')) #将停用词读取到列表里
    
    
    filename = "./noteevents1.csv"
    #read_data=pd.read_csv(filename, header=0, nrows=200)
    read_data=pd.read_csv(filename, header=0)
    raw_text = read_data.iloc[:,10].tolist()
    clean_text = map(cleantext, raw_text)
    print(type(clean_text))
    #sys.exit()
    read_data.iloc[:,10] = list(clean_text)
    read_data.to_csv("./noteevents_已经分词.csv", header=True)
    