from gensim.models import Word2Vec
from torch.utils import data
import os
import pandas as pd
import numpy as np
# from model1 import Bi_ATT_LSTM
from model import LSTM2, MULTModel_3, MULTModel_2, \
    Linear, HP, HP_DATA, MULTModel_4, Bi_ATT_LSTM, \
    FocalLoss, MULTModel_1, LSTM1
import torch
import torch.nn as nn
import torch.optim as optim
from utils import EICU_data
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    recall_score, precision_score, balanced_accuracy_score, \
    f1_score


class Model(nn.Module):
    """
    modality-wise LSTM
    """

    def __init__(self):
        super().__init__()
        self.lstm_1 = LSTM2(51, 200)
        self.lstm_2 = LSTM2(16, 200)
        self.lstm_3 = LSTM2(50, 200)
        self.fc_state = Linear(3, 10)
        self.fc1 = nn.Sequential(
            Linear(610, 610),
            nn.ReLU(),
            Linear(610, 610)
        )
        self.out_net = Linear(610, 2)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_state = inputs
        e_x_state = self.fc_state(x_state)  # 1, 1, dim
        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])
        h_3 = self.lstm_3([x_trt, t_trt[0]])
        x = torch.cat([h_1, h_2, h_3, e_x_state], dim=1)
        x = self.fc1(x) + x
        y = self.out_net(x)
        return torch.softmax(y, dim=1)


class Model0_1(nn.Module):
    """
    modality-wise LSTM, 1个模态
    """

    def __init__(self):
        super().__init__()
        self.lstm_1 = LSTM2(23, 100)
        self.fc_state = Linear(6, 20)
        self.fc = Linear(120, 2)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        h_1 = self.lstm_1([x_lab, t_lab[0]])
        y = self.fc(torch.cat([h_1, e_x_state], dim=1))
        return torch.softmax(y, dim=1)


class Model0_2(nn.Module):
    """
    modality-wise LSTM, 2个模态
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.lstm_1 = LSTM2(orig_d_1, d_1)
        self.lstm_2 = LSTM2(orig_d_2, d_2)
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc1 = nn.Sequential(
            Linear(d_1 + d_2 + HP_DATA.dim_x_state, d_1 + d_2 + HP_DATA.dim_x_state),
            nn.ReLU(),
            Linear(d_1 + d_2 + HP_DATA.dim_x_state, d_1 + d_2 + HP_DATA.dim_x_state)
        )
        self.out_net = Linear(d_1 + d_2 + HP_DATA.dim_x_state, HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim
        h_1 = self.lstm_1([x_lab, t_lab[0]])
        h_2 = self.lstm_2([x_vit, t_vit[0]])

        x = torch.cat([h_1, h_2, e_x_state], dim=1)
        x = self.fc1(x) + x
        y = self.out_net(x)

        return torch.softmax(y, dim=1)


class Model1(nn.Module):
    """
    cross-modality transformer
    """

    def __init__(self, HP_MT, HP_DATA):
        super().__init__()
        self.multimodal_transformer = MULTModel_3(HP_MT)
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(HP_DATA.dim_x_state + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0], \
                                                       x_trt, t_trt[0])

        y = self.fc(torch.cat([e_transformer, e_x_state], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model1_2(nn.Module):
    """
    cross-modality transformer
    只有两个模态
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.multimodal_transformer = MULTModel_2(HP_MT, orig_d_1, d_1, orig_d_2, d_2)
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(HP_DATA.dim_x_state + HP_MT.MulT_output_dim,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        e_transformer, _ = self.multimodal_transformer(x_lab, t_lab[0], \
                                                       x_vit, t_vit[0])

        y = self.fc(torch.cat([e_transformer, e_x_state], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3(nn.Module):
    """
    modality-wise transformer
    """

    def __init__(self, HP_MT, HP_DATA):
        super().__init__()
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.tf1 = MULTModel_1(HP_MT, HP_MT.orig_d_1, HP_MT.d_1)
        self.tf2 = MULTModel_1(HP_MT, HP_MT.orig_d_2, HP_MT.d_2)
        self.tf3 = MULTModel_1(HP_MT, HP_MT.orig_d_3, HP_MT.d_3)
        self.fc = Linear(HP_DATA.dim_x_state + HP_MT.d_1 + HP_MT.d_2 + \
                         HP_MT.d_3,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)
        e_trt = self.tf3(x_trt, t_trt)

        y = self.fc(torch.cat([e_lab, e_vit, e_trt, e_x_state], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model3_2(nn.Module):
    """
    modality-wise transformer
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1, orig_d_2, d_2):
        super().__init__()
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.tf2 = MULTModel_1(HP_MT, orig_d_2, d_2)
        self.fc = Linear(HP_DATA.dim_x_state + d_1 + d_2,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_vit, t_vit, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim

        e_lab = self.tf1(x_lab, t_lab)
        e_vit = self.tf2(x_vit, t_vit)

        y = self.fc(torch.cat([e_lab, e_vit, e_x_state], dim=1))

        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class Model5(nn.Module):
    """
    transformer 只有一个模态
    """

    def __init__(self, HP_MT, HP_DATA, orig_d_1, d_1):
        super().__init__()
        self.tf1 = MULTModel_1(HP_MT, orig_d_1, d_1)
        self.fc_state = Linear(HP_DATA.size_x_state, HP_DATA.dim_x_state)
        self.fc = Linear(HP_DATA.dim_x_state + d_1,
                         HP_MT.final_output_dim)

    def forward(self, inputs):
        x_lab, t_lab, \
        x_state = inputs

        e_x_state = self.fc_state(x_state)  # 1, 1, dim
        e_lab = self.tf1(x_lab, t_lab[0])

        y = self.fc(torch.cat([e_lab, e_x_state], dim=1))
        if torch.any(torch.isnan(y.detach().data)).item():
            raise ValueError("output nan")
        return torch.softmax(y, dim=1)


class HP_DATA1(HP_DATA):
    size_x_state = 3
    dim_x_state = 10


class HP1(HP):
    orig_d_1 = 51
    orig_d_2 = 16
    orig_d_3 = 50

    d_1 = 200
    d_2 = 200
    d_3 = 200

    num_heads = 4
    layers = 2

    MulT_output_dim = 400
    final_output_dim = 2


if __name__ == "__main__":

    modals = "ld"  # l-lab, v-vit, d-drug
    model_name = "all_tf"  # cmtf, mwlstm, mwtf, lstm, trsf, all_tf
    label_name = "discharge_death"
    Project = model_name + "-" + modals + "-" + label_name

    checkpoints_path = os.path.join(os.getcwd(), "checkpoints_eicu")
    train_records_path = os.path.join(os.getcwd(), "train_records_eicu")

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

    device = torch.device("cuda:0")  # "cpu"

    data = EICU_data(root="/home/luojiawei/eicu_R_work/all_unitstays/",
                     id_file="/home/luojiawei/eicu_R_work/id_files/train_id_" + label_name + ".csv")

    data_te = EICU_data(root="/home/luojiawei/eicu_R_work/all_unitstays/",
                        id_file="/home/luojiawei/eicu_R_work/id_files/test_id_" + label_name + ".csv")

    # model = Model()
    # model = Model0_3()
    # model = Model2()
    # model = Model1(HP1, HP_DATA1)
    model = Model1_2(HP1, HP_DATA1, 51, 200, 50, 200)
    # model = Model3(HP1, HP_DATA1)
    # model = Model5(HP1, HP_DATA1, HP1.orig_d_1, HP1.d_1)
    # model = Model1_1(HP1, HP_DATA1)
    model = model.to(device)

    # model_dict = model.state_dict()

    # pretrained_dict = torch.load(os.path.join(checkpoints_path, "param-mwlstm-lvdt-death24h-39.pth"), map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                    k in pretrained_dict and not k.find("lstm2.") != (-1)}
    # model_dict.update(pretrained_dict)

    # model_dict = torch.load(os.path.join(checkpoints_path, "param-mwtf-lvd-discharge_death-4.pth"), map_location=device)
    # model.load_state_dict(model_dict)
    # model.embed_text.weight.requires_grad = True

    # 冻结除了最后一层外的其他所有参数
    # freeze_param = ["fc.weight", "fc.bias"]
    # for name, param in model.named_parameters():
    #     if name not in freeze_param:
    #         param.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                          lr=0.01, momentum=0.9, weight_decay=0.0)

    BATCH_SIZE = 200
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

            pos_ratio = sum(list(map(lambda x: x[7][0], batches))) / len(batches)
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
                x_state, y = datas

                # yhat = model([free_text, t_free_text.to(device)])
                # yhat = model([free_text, t_free_text.to(device), x_state.to(device)])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_state.to(device)
                #               ])
                yhat = model([x_lab.to(device), t_lab.to(device), \
                              x_trt.to(device), t_trt.to(device), \
                              x_state.to(device)
                              ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               x_state.to(device)
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
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5)  # 梯度裁剪
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
            x_state, y = datas

            with torch.no_grad():
                # yhat = model([free_text, t_free_text.to(device)])
                # yhat = model([free_text, t_free_text.to(device), x_state.to(device)])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_state.to(device)
                #               ])
                yhat = model([x_lab.to(device), t_lab.to(device), \
                              x_trt.to(device), t_trt.to(device), \
                              x_state.to(device)
                              ])
                # yhat = model([x_lab.to(device), t_lab.to(device), \
                #               x_vit.to(device), t_vit.to(device), \
                #               x_trt.to(device), t_trt.to(device), \
                #               x_state.to(device)
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
