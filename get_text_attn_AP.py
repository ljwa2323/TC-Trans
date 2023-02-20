import sys

sys.path.append("/tmp/pycharm_project_74/")
sys.path.append("/tmp/pycharm_project_74/work_AP")

from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import torch
from utils import AP_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, \
    recall_score, precision_score, balanced_accuracy_score, \
    f1_score
from train_ap import Model1
from model import HP, HP_DATA

import json


class HP_DATA1(HP_DATA):
    size_x_state = 2
    dim_x_state = 10


class HP1(HP):
    orig_d_1 = 131
    orig_d_2 = 11
    orig_d_3 = 189
    orig_d_4 = 200

    d_1 = 100
    d_2 = 100
    d_3 = 100
    d_4 = 200

    num_heads = 1
    layers = 2

    MulT_output_dim = 300
    final_output_dim = 2


label_name = "discharge_death"
checkpoints_path = os.path.join("/tmp/pycharm_project_74/work_AP/", "checkpoints_AP")
device = torch.device("cuda:1")  # "cpu"
w2vmodel = Word2Vec.load(
    "/home/luojiawei/multimodal_model/data/用于多模态模型的新数据处理 20221015/word2vec_model/word2vec_model.model")

data_te = AP_data(root="/home/luojiawei/RL_AP/data/all_pids",
                  id_file="/home/luojiawei/RL_AP/data/id_file_test_death.csv",
                  wv=w2vmodel.wv)

model = Model1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
model = model.to(device)

model.load_state_dict(
    torch.load(os.path.join(checkpoints_path, "param-all_tf-lvd-discharge_death-43.pth"), map_location=device))

model = model.eval()

# np.random.seed(2000)
# ids = np.random.choice(data_te.len(), 100, replace=False).tolist()
# ids = np.random.choice(data_te.len(), 100, replace=False).tolist()
ids = list(range(data_te.len()))
batches, _ = data_te.get_data(ids)


def hook(module, input, output):
    features_in.append(input)
    features_out.append(output)
    return None


layer_names = []
for name, m in model.named_modules():
    layer_names.append(name)

txt = ["model." + layer_names[x] + ".register_forward_hook(hook)" for x in [242]]

for i in range(len(txt)):
    eval(txt[i])

root_path = "/home/luojiawei/MultTran/AP_output_files/"

print("开始输出信息")
for i in range(len(ids)):

    # i = 401

    cur_hamdid = str(data_te.all_pid[ids[i]])
    print("正在输出 {} 的信息 {} / {}".format(cur_hamdid, i, len(ids)))
    cur_path = os.path.join(root_path, cur_hamdid)

    if not os.path.exists(cur_path):
        os.mkdir(cur_path)

    features_in = []
    features_out = []
    datas = batches[i]

    x_lab, t_lab, \
    x_vit, t_vit, \
    x_trt, t_trt, \
    free_text, \
    x_state, y = datas
    # print(y)

    with torch.no_grad():
        yhat = model([x_lab.to(device), t_lab.to(device), \
                      x_vit.to(device), t_vit.to(device), \
                      x_trt.to(device), t_trt.to(device), \
                      free_text.to(device), \
                      x_state.to(device)])

    pred = str(yhat.cpu().detach().data.argmax(1).item())
    label = y.item()

    a = []
    index = free_text.cpu().detach().data.numpy().tolist()
    words = data_te.w2vmodel.untranslate_sentence1(index)
    wei = features_out[0][1].cpu().detach().data.numpy() / features_out[0][1].shape[0]
    wei_ma = wei.max()
    ratio = 0.5 / wei_ma
    wei = wei * ratio
    weights = wei.tolist()[0]
    weights = [round(x, 3) for x in weights]
    a.append({"words": words, "weights": weights, "prediction": [pred], "label": [label]})

    a1 = json.dumps(a,ensure_ascii=False)

    with open(os.path.join(cur_path, "attentions.json"), "w") as f:
        f.write(a1)