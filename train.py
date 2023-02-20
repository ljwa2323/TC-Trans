from gensim.models import Word2Vec
from torch.utils import data
import os
import pandas as pd
import numpy as np
# from model1 import Bi_ATT_LSTM
from model import LSTM2, MULTModel_3, MULTModel_2, \
    Linear, HP, HP_DATA, MULTModel_4, Bi_ATT_LSTM, \
    FocalLoss, MULTModel_1
import torch
import torch.nn as nn
import torch.optim as optim
from utils import MIMIC3_data
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    recall_score, precision_score, balanced_accuracy_score, \
    f1_score


class Model(nn.Module):
    """
    modality-wise LSTM
    """

    def __init__(self, wv):
        super().__init__()
        self.lstm_1 = LSTM2(83, 200)
        self.lstm_2 = LSTM2(34, 100)
        self.lstm_3 = LSTM2(50, 100)
        self.lstm2 = LSTM2(200, 200)
        self.fc_static = Linear(7, 20)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], 200 // 2)
        self.fc1 = nn.Sequential(
            Linear(620, 620),
            nn.ReLU(),
            Linear(620, 620)
        )
        self.out_net = Linear(620, 2)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim
        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])
        h_3 = self.lstm_3([x_trt, t_trt[0]])
        h_4 = self.lstm2([e_free_text, t_free_text[0]])
        x = torch.cat([h_1, h_2, h_3, h_4, e_x_static], dim=1)
        x = self.fc1(x) + x
        y = self.out_net(x)
        return torch.softmax(y, dim=1)


class Model0_3_1(nn.Module):
    """
    modality-wise LSTM
    三个模态，不包含 text 模态
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.lstm_2 = LSTM2(orig_d_2, d_2)
        self.lstm_3 = LSTM2(orig_d_3, d_3)
        self.fc_static = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(d_1 + d_2 + d_3, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])
        h_3 = self.lstm_3([x_trt, t_trt[0]])
        y = self.fc(torch.cat([h_1, h_2, h_3, e_x_static], dim=1))
        return torch.softmax(y, dim=1)


class Model0_3_2(nn.Module):
    """
    modality-wise LSTM
    三个模态，包含 text 模态
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.lstm_2 = LSTM2(orig_d_2, d_2)
        self.lstm_3 = LSTM2(orig_d_3, d_3)

        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], orig_d_3 // 2)

        self.fc_static = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc1 = nn.Sequential(
            Linear(d_1 + d_2 + d_3 + HP_DATA.dim_x_state, d_1 + d_2 + d_3 + HP_DATA.dim_x_state),
            nn.ReLU(),
            Linear(d_1 + d_2 + d_3 + HP_DATA.dim_x_state, d_1 + d_2 + d_3 + HP_DATA.dim_x_state)
        )
        self.out_net = Linear(d_1 + d_2 + d_3 + HP_DATA.dim_x_state, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        h_3 = self.lstm_3([e_free_text, t_free_text[0]])

        x = self.fc(torch.cat([h_1, h_2, h_3, e_x_static], dim=1))
        x = self.fc1(x) + x
        y = self.out_net(x)

        return torch.softmax(y, dim=1)


class Model0_2_1(nn.Module):
    """
    modality-wise LSTM
    2个模态，不包括 text
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.lstm_2 = LSTM2(orig_d_2, d_2)
        self.fc = Linear(d_1 + d_2 + HP_DATA.dim_x_state, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])
        y = self.fc(torch.cat([h_1, h_2, e_x_static], dim=1))
        return torch.softmax(y, dim=1)


class Model0_2_2(nn.Module):
    """
    modality-wise LSTM
    2个模态，包括 text
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.lstm_2 = LSTM2(orig_d_2, d_2)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], orig_d_2 // 2)

        self.fc_static = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc1 = nn.Sequential(
            Linear(d_1 + d_2 + HP_DATA.dim_x_state, d_1 + d_2 + HP_DATA.dim_x_state),
            nn.ReLU(),
            Linear(d_1 + d_2 + HP_DATA.dim_x_state, d_1 + d_2 + HP_DATA.dim_x_state)
        )
        self.out_net = Linear(d_1 + d_2 + HP_DATA.dim_x_state, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        h_2 = self.lstm_2([e_free_text, t_free_text[0]])

        x = self.fc(torch.cat([h_1, h_2, e_x_static], dim=1))
        x = self.fc1(x) + x
        y = self.out_net(x)

        return torch.softmax(y, dim=1)


class Model0_1(nn.Module):
    """
    modality-wise LSTM, 1个模态, 非文本
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.fc_static = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(d_1 + HP_DATA.dim_x_state, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        y = self.fc(torch.cat([h_1, e_x_static], dim=1))
        return torch.softmax(y, dim=1)


class Model1(nn.Module):
    """
    cross-modality transformer
    """

    def __init__(self, HP_MT, HP_DATA, wv):
        super().__init__()
        self.multimodal_transformer = MULTModel_3(HP_MT)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim + HP_MT.d_4,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))
        self.lstm2 = LSTM2(HP_MT.orig_d_4, HP_MT.d_4)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)
        h_text = self.lstm2([e_free_text, t_free_text[0]])

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       x_trt, t_trt[0])

        y = self.fc(torch.cat([e_transformer, e_x_static, h_text], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model2_1(nn.Module):
    """
    只有一个模态, 文本
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1):
        super().__init__()
        self.orig_d_1 = orig_d_1
        self.d_1 = d_1

        self.multimodal_transformer = MULTModel_1(HP_MT, orig_d_1, d_1)

        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)

        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入

        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))

        self.fc = Linear(self.d_1 + HP_DATA.dim_x_static,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)
        e_transformer = self.multimodal_transformer(e_free_text, t_free_text[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))

        return torch.softmax(y, dim=1)


class Model2_2(nn.Module):
    """
    只有一个模态, 非文本
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1):
        super().__init__()
        self.orig_d_1 = orig_d_1
        self.d_1 = d_1

        self.multimodal_transformer = MULTModel_1(HP_MT, orig_d_1, d_1)

        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)

        self.fc = Linear(self.d_1 + HP_DATA.dim_x_static,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_transformer = self.multimodal_transformer(x_lab, t_lab[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))

        return torch.softmax(y, dim=1)


class Model3(nn.Module):
    """
    modality-wise transformer
    """

    def __init__(self, HP_MT, HP_DATA, wv):
        super().__init__()
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.tf1 = MULTModel_1(HP_MT, HP_MT.orig_d_1, HP_MT.d_1)
        self.tf2 = MULTModel_1(HP_MT, HP_MT.orig_d_2, HP_MT.d_2)
        self.tf3 = MULTModel_1(HP_MT, HP_MT.orig_d_3, HP_MT.d_3)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.d_1 + HP_MT.d_2 + \
                         HP_MT.d_3 + HP_MT.d_4,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))
        self.lstm2 = LSTM2(HP_MT.orig_d_4, HP_MT.d_4)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)
        h_text = self.lstm2([e_free_text, t_free_text[0]])

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)
        e_trt = self.tf3(x_trt, t_trt)

        y = self.fc(torch.cat([e_lab, e_vit, e_trt, e_x_static, h_text], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3_3_1(nn.Module):
    """
    modality-wise transformer - 3个动态源，包含文本
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.tf2 = MULTModel_1(HP_MT, orig_d_2, d_2)
        self.fc = Linear(HP_DATA.dim_x_static + d_1 + d_2 + \
                         HP_MT.d_4,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))
        self.lstm2 = LSTM2(HP_MT.orig_d_4, HP_MT.d_4)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)
        h_text = self.lstm2([e_free_text, t_free_text[0]])

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)

        y = self.fc(torch.cat([e_lab, e_vit, e_x_static, h_text], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3_3_2(nn.Module):
    """
    modality-wise transformer - 3个动态源，不包含文本
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3):
        super().__init__()
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.tf2 = MULTModel_1(HP_MT, orig_d_2, d_2)
        self.tf3 = MULTModel_1(HP_MT, orig_d_3, d_3)
        self.fc = Linear(HP_DATA.dim_x_static + d_1 + d_2 + d_3,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)
        e_trt = self.tf3(x_trt, t_trt)

        y = self.fc(torch.cat([e_lab, e_vit, e_trt, e_x_static], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3_2_1(nn.Module):
    """
    modality-wise transformer - 2个动态源，包含文本
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1):
        super().__init__()
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.fc = Linear(HP_DATA.dim_x_static + d_1 + \
                         HP_MT.d_4,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))
        self.lstm2 = LSTM2(HP_MT.orig_d_4, HP_MT.d_4)

    def forward(self, inputs):
        x_lab, t_lab, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:500] if free_text[i].shape[0] >= 500 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)
        h_text = self.lstm2([e_free_text, t_free_text[0]])

        e_lab = self.tf1(x_lab, t_lab)

        y = self.fc(torch.cat([e_lab, e_x_static, h_text], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3_2_2(nn.Module):
    """
    modality-wise transformer - 2个动态源，包含文本
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.tf2 = MULTModel_1(HP_MT, orig_d_2, d_2)
        self.fc = Linear(HP_DATA.dim_x_static + d_1 + d_2,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)

        y = self.fc(torch.cat([e_lab, e_vit, e_x_static], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model4(nn.Module):
    """
    cross-modality transformer
    4 modalities
    """

    def __init__(self, HP_MT, HP_DATA, wv):
        super().__init__()
        self.multimodal_transformer = MULTModel_4(HP_MT)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:2000] if free_text[i].shape[0] >= 2000 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       x_trt, t_trt[0], \
                                                       e_free_text, t_free_text[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model1_2_1(nn.Module):
    """
    transformer 2 个模态 不包括文本模态
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.multimodal_transformer = MULTModel_2(HP_MT, orig_d_1, d_1, orig_d_2, d_2)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_static = inputs

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0])

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model1_2_2(nn.Module):
    """
    transformer 2 个模态 其中一个必须是文本
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.multimodal_transformer = MULTModel_2(HP_MT, orig_d_1, d_1, orig_d_2, d_2)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)

        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))

    def forward(self, inputs):
        x_lab, t_lab, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       e_free_text, t_free_text[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model6_1(nn.Module):
    """
    cross-modality transformer
    3 个模态, 不包含 text 模态
    """

    def __init__(self, HP_MT, HP_DATA):
        super().__init__()
        self.multimodal_transformer = MULTModel_3(HP_MT)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       x_trt, t_trt[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))
        return torch.softmax(y, dim=1)


class Model6_2(nn.Module):
    """
    cross-modality transformer
    3 个模态, 包含 text 模态
    """

    def __init__(self, HP_MT, HP_DATA, wv, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3):
        super().__init__()
        self.multimodal_transformer = MULTModel_3(HP_MT, orig_d_1, d_1, orig_d_2, d_2, orig_d_3, d_3)
        self.fc_static = Linear(HP_DATA.size_x_static, HP_DATA.dim_x_static)
        self.fc = Linear(HP_DATA.dim_x_static + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)
        self.embed_text = nn.Embedding.from_pretrained(wv)  # 词嵌入
        self.lstm = Bi_ATT_LSTM(self.embed_text.weight.shape[1], int(HP_MT.orig_d_4 / 2))

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        free_text, t_free_text, \
        x_static = inputs

        e_x_static = self.fc_static(x_static)  # 1, 1, dim
        e_free_text = []
        for i in range(len(free_text)):
            # word(time), dim -->1, time, dim --> 1, dim
            # cur_text = free_text[i][:2000] if free_text[i].shape[0] >= 2000 else free_text[i]
            cur_text = free_text[i]
            lstm_out, _ = self.lstm(self.embed_text(cur_text).float().unsqueeze(0))
            e_free_text.append(lstm_out)
        e_free_text = torch.stack(e_free_text, dim=1).float()  # (1, time, dim)

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       e_free_text, t_free_text[0])

        y = self.fc(torch.cat([e_transformer, e_x_static], dim=1))
        return torch.softmax(y, dim=1)


class HP_DATA1(HP_DATA):
    size_x_static = 7
    dim_x_static = 20


class HP1(HP):
    orig_d_1 = 83
    orig_d_2 = 34
    orig_d_3 = 50
    orig_d_4 = 100

    d_1 = 100
    d_2 = 100
    d_3 = 100
    d_4 = 100

    num_heads = 4
    layers = 2

    MulT_output_dim = 400
    final_output_dim = 2


if __name__ == "__main__":

    modals = "vdt"  # l-lab, v-vit, d-drug, t-text
    model_name = "mwtf"  # cmtf, mwlstm, mwtf, lstm, trsf, all_tf
    label_name = "death24h"
    Project = model_name + "-" + modals + "-" + label_name

    checkpoints_path = os.path.join(os.getcwd(), "checkpoints")
    train_records_path = os.path.join(os.getcwd(), "train_records")

    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
        print(checkpoints_path + "  已创建")

    if not os.path.exists(train_records_path):
        os.mkdir(train_records_path)
        print(train_records_path + "  已创建")

    record_file = os.path.join(train_records_path, "train_record-" + Project + ".txt")
    # if os.path.exists(record_file):
    #     os.remove(record_file)
    with open(record_file, "a") as f:
        f.write("Mode\tEpoch\tN\tloss\tAUC\tACC\tR\tP\tF1\tpr\n")

    device = torch.device("cuda:2")  # "cpu"

    w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")

    print("word2vec load succuess.")
    data = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                       id_file="/home/luojiawei/mimic3_R_work/id_files/train_id_death24h.csv",
                       wv=w2vmodel.wv)

    data_te = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                          id_file="/home/luojiawei/mimic3_R_work/id_files/test_id_death24h.csv",
                          wv=w2vmodel.wv)

    # model = Model(wv=torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model0_3()
    # model = Model2(wv=torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model3(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
    model = Model3_3_1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float(),
                       34, 100, 50, 100)
    # model = Model3_3_2(HP1, HP_DATA1, 83,100, 34, 100, 50, 100)
    # model = Model4(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model1_2_2(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float(), 83, 100, 100, 100)
    # model = Model2_1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float(), 100, 100)
    # model = Model2_2(HP1, HP_DATA1, 34, 100)
    # model = Model1_1(HP1, HP_DATA1)
    model = model.to(device)

    # model_dict = model.state_dict()

    # pretrained_dict = torch.load(os.path.join(checkpoints_path, "param-all_tf-lvdt-death24h-19.pth"), map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                    k in pretrained_dict and not k.find("lstm2.") != (-1)}
    # model_dict.update(pretrained_dict)

    # model_dict = torch.load(os.path.join(checkpoints_path, "param-all_tf-lvdt-death24h-18.pth"), map_location=device)
    # model.load_state_dict(model_dict)
    # model.embed_text.weight.requires_grad = True

    # 冻结除了最后一层外的其他所有参数
    # freeze_param = ["fc.weight", "fc.bias"]
    # for name, param in model.named_parameters():
    #     if name not in freeze_param:
    #         param.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                          lr=0.005, momentum=0.9, weight_decay=0.001)

    BATCH_SIZE = 150
    STEP_SIZE = data.len() // BATCH_SIZE if data.len() % BATCH_SIZE == 0 else data.len() // BATCH_SIZE + 1
    MAX_iter = STEP_SIZE * 1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=MAX_iter,  # 学习率衰减周期
                                             gamma=0.99)
    epoch_start = 1
    EPOCH = 40
    print("开始训练")
    for epoch in range(epoch_start, EPOCH):

        model.train()

        loader = data.iterate_batch(size=BATCH_SIZE, shuffle=True)

        for n, (batches, ids) in enumerate(loader):

            if n >= MAX_iter:
                break

            if (len(batches) == 0):
                continue

            loss = 0.0

            pos_ratio = sum(list(map(lambda x: x[9][0], batches))) / len(batches)
            weight = torch.tensor([1 / (1 - pos_ratio + 1e-6), 1 / (pos_ratio + 1e-6)], dtype=torch.float32)
            weight = torch.clamp(weight, 1.0, 200.0)

            print("本轮的weight是:{}".format(weight.data))
            Loss_fun = nn.CrossEntropyLoss(weight=weight.to(device))
            # Loss_fun = FocalLoss(weight=weight.to(device))

            optimizer.zero_grad()
            print("开始前向运算")

            y_true = []
            y_pred = []
            for i in range(len(batches)):
                # print(data.all_hadm_id[ids[i]])
                # if i % 5 == 0:
                #     print(i)
                datas = batches[i]

                x_lab, t_lab, \
                x_vit, t_vit, \
                x_trt, t_trt, \
                free_text, t_free_text, \
                x_static, y = datas

                free_text = [free_text[i].to(device) for i in range(len(free_text))]

                # yhat = model([free_text, t_free_text.to(device)])
                # yhat = model([free_text, t_free_text.to(device), x_static.to(device)])
                # yhat = model([x_vit.to(device), t_vit.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])
                yhat = model([x_vit.to(device), t_vit.to(device), \
                              x_trt.to(device), t_trt.to(device), \
                              free_text, t_free_text.to(device), \
                              x_static.to(device)
                              ])

                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])

                # yhat: batch=1, 100 | y: batch=1, 100
                cur_loss = Loss_fun(yhat, y.to(device))
                loss += cur_loss

                y_pred.append(yhat.cpu().data.numpy().reshape(1, -1))
                y_true.append(y.numpy().reshape(1, -1))

            y_true = np.concatenate(y_true, axis=0)  # batch, 1
            y_pred = np.concatenate(y_pred, axis=0)  # batch, 2
            y_pred1 = y_pred.argmax(axis=1)

            loss /= len(batches)
            print("开始梯度反向传播,更新参数")
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=10)  # 梯度裁剪
            optimizer.step()
            lr_scheduler.step()  # 学习率更新

            print("Epoch:{} - {} || 学习率: {:.6f} ||  loss: {:.3f}".format(epoch, n, \
                                                                         lr_scheduler.get_last_lr()[0], \
                                                                         loss.cpu().detach().data))
            print(confusion_matrix(y_true[:, 0], y_pred1))
            AUC = np.round(roc_auc_score(y_true[:, 0], y_pred[:, 1]), 3)
            recall = np.round(recall_score(y_true[:, 0], y_pred1), 3)
            precision = np.round(precision_score(y_true[:, 0], y_pred1), 3)
            bal_acc = np.round(balanced_accuracy_score(y_true[:, 0], y_pred1), 3)
            f1 = np.round(f1_score(y_true[:, 0], y_pred1), 3)
            pos_ratio = np.round(y_true[:, 0].sum() / y_true.shape[0], 3)

            ds = {"AUC": [AUC],
                  "ACC": [bal_acc],
                  "recall": [recall],
                  "precision": [precision],
                  "f1_score": [f1],
                  "pos_ratio": [pos_ratio]}

            ds = pd.DataFrame(ds)
            print(ds)

            with open(record_file, "a") as f:
                f.write("train\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(epoch, n, \
                                                                                                         loss.cpu().detach().data, \
                                                                                                         AUC, bal_acc, \
                                                                                                         recall,
                                                                                                         precision, \
                                                                                                         f1, pos_ratio))

        if data_te.len() >= 1000:
            ids = np.random.choice(data_te.len(), 1000, replace=False).tolist()
        else:
            ids = np.arange(data_te.len())

        batches, _ = data_te.get_data(ids)

        y_true = []
        y_pred = []

        model.eval()
        print("开始测试")
        for i in range(len(batches)):
            datas = batches[i]

            x_lab, t_lab, \
            x_vit, t_vit, \
            x_trt, t_trt, \
            free_text, t_free_text, \
            x_static, y = datas

            free_text = [free_text[i].to(device) for i in range(len(free_text))]
            with torch.no_grad():
                # yhat = model([free_text, t_free_text.to(device)])
                # yhat = model([free_text, t_free_text.to(device), x_static.to(device)])
                # yhat = model([x_vit.to(device), t_vit.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])
                yhat = model([x_vit.to(device), t_vit.to(device), \
                              x_trt.to(device), t_trt.to(device), \
                              free_text, t_free_text.to(device), \
                              x_static.to(device)
                              ])
                # yhat = model([x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               x_static.to(device)
                #               ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               free_text, t_free_text.to(device), \
                #               x_static.to(device)
                #               ])

            y_pred.append(yhat.cpu().data.numpy().reshape(1, -1))
            y_true.append(y.numpy().reshape(1, -1))

        y_true = np.concatenate(y_true, axis=0)  # batch, 1
        y_pred = np.concatenate(y_pred, axis=0)  # batch, 2
        y_pred1 = y_pred.argmax(1)

        AUC = np.round(roc_auc_score(y_true[:, 0], y_pred[:, 1]), 2)
        recall = np.round(recall_score(y_true[:, 0], y_pred1), 2)
        precision = np.round(precision_score(y_true[:, 0], y_pred1), 2)
        bal_acc = np.round(balanced_accuracy_score(y_true[:, 0], y_pred1), 2)
        f1 = np.round(f1_score(y_true[:, 0], y_pred1), 2)
        pos_ratio = np.round(y_true[:, 0].sum() / y_true.shape[0], 2)

        print("test\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(epoch, "NA", "NA", \
                                                                                          AUC, bal_acc, \
                                                                                          recall, precision, \
                                                                                          f1, pos_ratio))

        with open(record_file, "a") as f:
            f.write("test\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(epoch, "NA", "NA", \
                                                                                                AUC, bal_acc, \
                                                                                                recall, precision, \
                                                                                                f1, pos_ratio))
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_path, "param-" + Project + "-" + str(epoch) + ".pth"))
        print(
            os.path.join(checkpoints_path, "param-" + Project + "-" + str(epoch) + ".pth") + "  已保存")
