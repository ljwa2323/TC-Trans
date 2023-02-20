import sys

sys.path.append("/tmp/pycharm_project_74/")

from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import torch
from utils import MIMIC3_data

from train import Model4
import json

from model import HP, HP_DATA


class HP_DATA1(HP_DATA):
    size_x_static = 7
    dim_x_static = 20


class HP1(HP):
    orig_d_1 = 83
    orig_d_2 = 34
    orig_d_3 = 50
    orig_d_4 = 200

    d_1 = 200
    d_2 = 200
    d_3 = 200
    d_4 = 200

    num_heads = 1
    layers = 2

    MulT_output_dim = 400
    final_output_dim = 2


label_name = "death24h"
checkpoints_path = os.path.join("/tmp/pycharm_project_74", "checkpoints")
device = torch.device("cpu")  # "cpu"

w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")

data_te = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                      id_file="/home/luojiawei/mimic3_R_work/id_files/test_id_" + label_name + ".csv",
                      wv=w2vmodel.wv)

model = Model4(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
model = model.to(device)

model.load_state_dict(
    torch.load(os.path.join(checkpoints_path, "param-all_tf-lvdt-death24h-18.pth"), map_location=device))

model = model.eval()

np.random.seed(2000)
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

txt = ["model." + layer_names[x] + ".register_forward_hook(hook)" for x in [412]]

for i in range(len(txt)):
    eval(txt[i])

root_path = "/home/luojiawei/MultTran/mimic3_output_files/"

print("开始输出信息")
for j1 in range(len(ids)):
    # j1 = 401

    cur_hamdid = str(data_te.all_hadm_id[ids[j1]])
    print("正在输出 {} 的信息 {} / {}".format(cur_hamdid, j1, len(ids)))
    cur_path = os.path.join(root_path, cur_hamdid)

    if not os.path.exists(cur_path):
        os.mkdir(cur_path)

    features_in = []
    features_out = []

    datas = batches[j1]

    x_lab, t_lab, \
    x_vit, t_vit, \
    x_trt, t_trt, \
    free_text, t_free_text, \
    x_state, y = datas
    # print(y)

    free_text = [free_text[i].to(device) for i in range(len(free_text))]

    model = model.eval()

    with torch.no_grad():
        yhat = model([x_lab.to(device), t_lab.to(device), \
                      x_vit.to(device), t_vit.to(device), \
                      x_trt.to(device), t_trt.to(device), \
                      free_text, t_free_text.to(device), \
                      x_state.to(device)])

    pred = str(yhat.cpu().detach().data.argmax(1).item())

    label = str(y.item())

    features_in.__len__()

    a = []

    for i in range(features_in.__len__()):
        index = free_text[i].cpu().detach().data.numpy().tolist()
        words = data_te.w2vmodel.untranslate_sentence1(index)
        wei = features_out[i][1].cpu().detach().data.numpy() / features_out[i][1].shape[0]
        wei_ma = wei.max()
        ratio = 0.5 / wei_ma
        wei = wei * ratio
        weights = wei.tolist()[0]
        weights = [round(x, 3) for x in weights]
        a.append({"words": words, "weights": weights, "prediction": [pred], "label": [label]})

    # print(a.__len__())
    # print(t_free_text.shape)
    a1 = json.dumps(a)

    with open(os.path.join(cur_path, "attentions.json"), "w") as f:
        f.write(a1)
