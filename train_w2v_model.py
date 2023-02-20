import pandas as pd
from gensim.models import Word2Vec
# import sys

print("开始读取数据")
ds = pd.read_csv("/home/luojiawei/mimic3_R_work/noteevents_已经分词2.csv", header=0)
# print(ds.dtypes)
# print(ds['text1'].dtype)


print("读取数据完成")

corups = list(map(lambda x: str(x).split(" "), ds.iloc[:, 12]))
# print(corups[0])
# sys.exit()
print("开始训练模型")
model = Word2Vec(sentences=corups,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 workers=4,
                 sg=1,
                 hs=1)

print("开始保存模型")
model.save("/home/luojiawei/mimic3_R_work/word2vec_model.model")
print("模型保存成功")
