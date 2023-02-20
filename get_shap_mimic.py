import sys

sys.path.append("/tmp/pycharm_project_74/")

from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import torch
from utils import MIMIC3_data
from model import LSTM2, MULTModel_3, MULTModel_2, \
    Linear, HP, HP_DATA, MULTModel_4, Bi_ATT_LSTM, \
    FocalLoss, MULTModel_1
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, \
    recall_score, precision_score, balanced_accuracy_score, \
    f1_score
from train import Model, Model0_1, Model1, Model2, Model1_1, HP1, HP_DATA1, Model3, Model4
import pandas as pd

import os

HP1.MulT_output_dim = 400
HP1.d_1 = 200
HP1.d_2 = 200
HP1.d_3 = 200
HP1.d_4 = 200
HP1.orig_d_4 = 200
HP1.layers = 2

class Model4_1(nn.Module):
    """
    cross-modality transformer
    4 modalities
    """

    def __init__(self, HP_MT, HP_DATA):
        super().__init__()
        self.multimodal_transformer = MULTModel_4(HP_MT)
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(HP_DATA.dim_x_state + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        e_free_text, t_free_text, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       x_trt, t_trt[0], \
                                                       e_free_text, t_free_text[0])

        y = self.fc(torch.cat([e_transformer, e_x_state], dim=1))
        # if torch.any(torch.isnan(y.detach().data)).item():
        #     raise ValueError("output nan")
        return torch.softmax(y, dim=1)


label_name = "death24h"
checkpoints_path = os.path.join("/tmp/pycharm_project_74", "checkpoints")
device = torch.device("cpu")  # "cpu"

w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")

data_te = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                      id_file="/home/luojiawei/mimic3_R_work/id_files/test_id_" + label_name + ".csv",
                      wv=w2vmodel.wv,
                      label_name=label_name)

model = Model4(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
model = model.to(device)
model = model.eval()

model1 = Model4(HP1, HP_DATA1)
model1 = model1.to(device)
model1 = model1.eval()

model_dict = model.state_dict()
model_dict1 = model1.state_dict()

pretrained_dict = torch.load(os.path.join(checkpoints_path, "param-all_tf-lvdt-death24h-19.pth"), map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in pretrained_dict and not k.find("lstm2.") != (-1)}
pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict1}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model_dict1.update(pretrained_dict1)
model1.load_state_dict(model_dict1)

np.random.seed(2000)
ids = np.random.choice(data_te.len(), 100, replace=False).tolist()
# ids = list(range(data_te.len()))
batches, _ = data_te.get_data(ids)