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

txt = ["model." + layer_names[x] + ".register_forward_hook(hook)" for x in
       [6, 29, 52, 75, 98, 121, 144, 167, 190, 213, 236, 259]]

for i in range(len(txt)):
    eval(txt[i])

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

    with torch.no_grad():
        yhat = model([x_lab.to(device), t_lab.to(device), \
                      x_vit.to(device), t_vit.to(device), \
                      x_trt.to(device), t_trt.to(device), \
                      free_text, t_free_text.to(device), \
                      x_state.to(device)])

    yhat = yhat.cpu().data.numpy()
    yhat = pd.DataFrame(yhat)
    yhat.to_csv(os.path.join(cur_path, "./yhat.csv"), index=False)

    modal1 = pd.DataFrame(x_lab[0].cpu().detach().data.numpy())
    t_modal1 = pd.DataFrame(t_lab.cpu().detach().data.numpy().reshape(-1, 1))
    # print(t_modal1.shape)

    modal2 = pd.DataFrame(x_vit[0].cpu().detach().data.numpy())
    t_modal2 = pd.DataFrame(t_vit.cpu().detach().data.numpy().reshape(-1, 1))
    # print(t_modal2.shape)

    modal3 = pd.DataFrame(x_trt[0].cpu().detach().data.numpy())
    t_modal3 = pd.DataFrame(t_trt.cpu().detach().data.numpy().reshape(-1, 1))

    t_modal4 = pd.DataFrame(t_free_text.cpu().detach().data.numpy().reshape(-1, 1))

    attn1_2 = pd.DataFrame(np.round(features_out[0][1][0].cpu().detach().data.numpy()[0], 3))

    attn1_3 = pd.DataFrame(np.round(features_out[1][1][0].cpu().detach().data.numpy()[0], 3))

    attn1_4 = pd.DataFrame(np.round(features_out[2][1][0].cpu().detach().data.numpy()[0], 3))

    attn2_1 = pd.DataFrame(np.round(features_out[3][1][0].cpu().detach().data.numpy()[0], 3))

    attn2_3 = pd.DataFrame(np.round(features_out[4][1][0].cpu().detach().data.numpy()[0], 3))

    attn2_4 = pd.DataFrame(np.round(features_out[5][1][0].cpu().detach().data.numpy()[0], 3))

    attn3_1 = pd.DataFrame(np.round(features_out[6][1][0].cpu().detach().data.numpy()[0], 3))

    attn3_2 = pd.DataFrame(np.round(features_out[7][1][0].cpu().detach().data.numpy()[0], 3))

    attn3_4 = pd.DataFrame(np.round(features_out[8][1][0].cpu().detach().data.numpy()[0], 3))

    attn4_1 = pd.DataFrame(np.round(features_out[9][1][0].cpu().detach().data.numpy()[0], 3))

    attn4_2 = pd.DataFrame(np.round(features_out[10][1][0].cpu().detach().data.numpy()[0], 3))

    attn4_3 = pd.DataFrame(np.round(features_out[11][1][0].cpu().detach().data.numpy()[0], 3))

    modal1.to_csv(os.path.join(cur_path, "./modal1.csv"), index=False)
    t_modal1.to_csv(os.path.join(cur_path, "./t_modal1.csv"), index=False)
    modal2.to_csv(os.path.join(cur_path, "./modal2.csv"), index=False)
    t_modal2.to_csv(os.path.join(cur_path, "./t_modal2.csv"), index=False)
    modal3.to_csv(os.path.join(cur_path, "./modal3.csv"), index=False)
    t_modal3.to_csv(os.path.join(cur_path, "./t_modal3.csv"), index=False)
    t_modal4.to_csv(os.path.join(cur_path, "./t_modal4.csv"), index=False)

    attn1_2.to_csv(os.path.join(cur_path, "./attn1_2.csv"), index=False)
    attn1_3.to_csv(os.path.join(cur_path, "./attn1_3.csv"), index=False)
    attn1_4.to_csv(os.path.join(cur_path, "./attn1_4.csv"), index=False)

    attn2_1.to_csv(os.path.join(cur_path, "./attn2_1.csv"), index=False)
    attn2_3.to_csv(os.path.join(cur_path, "./attn2_3.csv"), index=False)
    attn2_4.to_csv(os.path.join(cur_path, "./attn2_4.csv"), index=False)

    attn3_1.to_csv(os.path.join(cur_path, "./attn3_1.csv"), index=False)
    attn3_2.to_csv(os.path.join(cur_path, "./attn3_2.csv"), index=False)
    attn3_4.to_csv(os.path.join(cur_path, "./attn3_4.csv"), index=False)

    attn4_1.to_csv(os.path.join(cur_path, "./attn4_1.csv"), index=False)
    attn4_2.to_csv(os.path.join(cur_path, "./attn4_2.csv"), index=False)
    attn4_3.to_csv(os.path.join(cur_path, "./attn4_3.csv"), index=False)
