import sys

sys.path.append("/tmp/pycharm_project_74/")

from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import torch
from utils import MIMIC3_data

from train import Model4
import pandas as pd

from model import HP, HP_DATA

import os


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

# model = Model1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
model = Model4(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
model = model.to(device)

model.load_state_dict(
    torch.load(os.path.join(checkpoints_path, "param-all_tf-lvdt-death24h-18.pth"), map_location=device))
model = model.eval()

np.random.seed(2000)
# ids = np.random.choice(data_te.len(), 100, replace=False).tolist()
ids = list(range(data_te.len()))
batches, _ = data_te.get_data(ids)

root_path = "/home/luojiawei/MultTran/mimic3_output_files/"

print("开始输出信息")
for i in range(len(ids)):

    # i = 401

    cur_hamdid = str(data_te.all_hadm_id[ids[i]])
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
    free_text, t_free_text, \
    x_state, y = datas
    # print(y)

    free_text = [free_text[i].to(device) for i in range(len(free_text))]

    x_lab = x_lab / 2
    x_vit = x_vit / 2
    x_trt = x_trt / 2

    x_lab.requires_grad = True
    x_vit.requires_grad = True
    x_trt.requires_grad = True
    x_state.requires_grad = True

    yhat = model([x_lab.to(device), t_lab.to(device), \
                  x_vit.to(device), t_vit.to(device), \
                  x_trt.to(device), t_trt.to(device), \
                  free_text, t_free_text.to(device), \
                  x_state.to(device)])

    grads = torch.autograd.grad(outputs=yhat[0, 1],
                                inputs=[x_lab, x_vit, x_trt],
                                retain_graph=False)

    x_lab_grad = grads[0] * x_lab
    x_vit_grad = grads[1] * x_vit
    x_trt_grad = grads[2] * x_trt

    yhat = yhat.cpu().data.numpy()

    x_lab_grad = x_lab_grad[0].data.numpy()
    x_vit_grad = x_vit_grad[0].data.numpy()
    x_trt_grad = x_trt_grad[0].data.numpy()

    x_lab_grad = pd.DataFrame(x_lab_grad)
    x_vit_grad = pd.DataFrame(x_vit_grad)
    x_trt_grad = pd.DataFrame(x_trt_grad)
    yhat = pd.DataFrame(yhat)

    x_lab_grad.to_csv(os.path.join(cur_path, "./grad_modal1.csv"), index=False)
    x_vit_grad.to_csv(os.path.join(cur_path, "./grad_modal2.csv"), index=False)
    x_trt_grad.to_csv(os.path.join(cur_path, "./grad_modal3.csv"), index=False)
    yhat.to_csv(os.path.join(cur_path, "./yhat.csv"), index=False)

