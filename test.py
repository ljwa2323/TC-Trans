from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import torch
from utils import MIMIC3_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, \
    recall_score, precision_score, balanced_accuracy_score, \
    f1_score
from train import Model, Model0_1, Model1, Model2, Model1_1, HP1, HP_DATA1

if __name__ == "__main__":

    label_name = "death24h"

    checkpoints_path = os.path.join(os.getcwd(), "checkpoints")

    device = torch.device("cuda:1")  # "cpu"

    w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")

    print("word2vec load succuess.")
    data = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                       id_file="/home/luojiawei/mimic3_R_work/id_files/test_id_" + label_name + ".csv",
                       wv=w2vmodel.wv,
                       label_name=label_name)

    # model = Model2(torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model1(HP1, HP_DATA1, torch.from_numpy(w2vmodel.wv.vectors).float())
    # model = Model0_1()
    model = Model1_1(HP1, HP_DATA1)
    model = model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(checkpoints_path, "param-mwlstm-lvd-death24h-0-5.pth"), map_location=device))

    ids = np.random.choice(data.len(), 200, replace=False).tolist()

    batches, _ = data.get_data(ids)

    y_true = []
    y_pred = []

    model.eval()
    print("开始测试")
    for i in range(len(batches)):

        if i % 5 == 0:
            print(i)

        datas = batches[i]

        x_lab, t_lab, \
        x_vit, t_vit, \
        x_trt, t_trt, \
        free_text, t_free_text, \
        x_state, y = datas
        # y: batch=1, 100

        free_text = [free_text[i].to(device) for i in range(len(free_text))]
        with torch.no_grad():

            # yhat: batch=1, 100
            # yhat = model([free_text, t_free_text.to(device)])
            yhat = model([x_lab.to(device), t_lab.to(device), \
                          x_vit.to(device), t_vit.to(device), \
                          x_trt.to(device), t_trt.to(device), \
                          x_state.to(device)])
            # yhat = model([x_lab.to(device), t_lab.to(device), \
            #               x_vit.to(device), t_vit.to(device), \
            #               x_trt.to(device), t_trt.to(device), \
            #               free_text, t_free_text.to(device), \
            #               x_state.to(device)])

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

    ds = {"AUC": [AUC],
          "ACC": [bal_acc],
          "recall": [recall],
          "precision": [precision],
          "f1_score": [f1],
          "pos_ratio": [pos_ratio]}

    ds = pd.DataFrame(ds)
    print(ds)
    print(classification_report(y_true[:, 0], y_pred1))
    print(confusion_matrix(y_true[:, 0], y_pred1))
