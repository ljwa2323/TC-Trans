from gensim.models import Word2Vec
from torch.utils import data
import os
import pandas as pd
import numpy as np
from model import HP, HP_DATA
import torch
import torch.nn as nn


class W2Vmodel(object):

    def __init__(self, wv):
        self.wv = wv
        self.vocab = wv.key_to_index

    def translate_sentence(self, sentence):
        return list(map(self.wv.get_vectors, sentence))

    def translate_sentence1(self, sentence):
        return list(map(lambda x: self.wv.key_to_index[x], sentence))

    def untranslate_sentence1(self, sentence):
        return list(map(lambda x: self.wv.index_to_key[x], sentence))


class MIMIC3_data(object):

    def __init__(self, root, id_file, wv):
        """
        初始化的时候，需要直接输入不同模态的所有数据
        :param root: 包含所有病人文件夹的根文件夹路径

        """
        super().__init__()
        import pandas as pd
        self.root = root
        ds = pd.read_csv(id_file, header=0)
        self.all_hadm_id = ds.loc[:, "id"].tolist()
        self.w2vmodel = W2Vmodel(wv)
        print('一共有{}人次'.format(self.len()))

    def len(self):
        return (len(self.all_hadm_id))

    def get_item(self, ind):
        folder_path = os.path.join(self.root, str(self.all_hadm_id[ind]))
        admissions = pd.read_csv(os.path.join(folder_path, "admissions.csv"), header=0)

        admissions1 = admissions.iloc[0, :].tolist()
        x_static = np.array([admissions1[i] for i in [5, 6, 7, 8, 9, 11, 17]]).astype("float32").reshape(1, -1)

        # 诊断标签
        y = np.array([admissions.iloc[0, 20]]).astype("int64")

        chartevents_n = pd.read_csv(os.path.join(folder_path, "chartevents1_n.csv"), header=0)

        if chartevents_n.shape[0] == 0:
            return None

        x_vit = np.asarray(chartevents_n.iloc[:, 1:]).astype("float32").reshape(1, chartevents_n.shape[0], -1)
        t_vit = np.asarray(chartevents_n.iloc[:, 0]).astype("float32").reshape(1, chartevents_n.shape[0])

        labevents_n = pd.read_csv(os.path.join(folder_path, "labevents2_n.csv"), header=0)

        if labevents_n.shape[0] == 0:
            return None

        x_lab = np.asarray(labevents_n.iloc[:, 1:]).astype("float32").reshape(1, labevents_n.shape[0], -1)
        t_lab = np.asarray(labevents_n.iloc[:, 0]).astype("float32").reshape(1, labevents_n.shape[0])

        prescriptions = pd.read_csv(os.path.join(folder_path, "prescriptions.csv"), header=0)
        if prescriptions.shape[0] == 0:
            return None

        x_trt = np.asarray(prescriptions.iloc[:, 1:]).astype("float32").reshape(1, prescriptions.shape[0], -1)
        t_trt = np.asarray(prescriptions.iloc[:, 0]).astype("float32").reshape(1, prescriptions.shape[0])

        noteevents = pd.read_csv(os.path.join(folder_path, "noteevents.csv"), header=0)
        noteevents = noteevents.loc[:, ["text1", "time"]]
        noteevents = noteevents.sort_values(by="time", ascending=True)

        if noteevents.shape[0] == 0:
            return None

        free_text = []
        index = []
        for i in range(noteevents.shape[0]):
            if noteevents.iloc[i, 0] is np.NAN:
                continue
            index.append(i)
            free_text.append(
                np.array(self.w2vmodel.translate_sentence1(noteevents.iloc[i, 0].split(" "))).astype("int64").reshape(
                    -1, ))

        t_free_text = np.asarray(noteevents.iloc[index, 1]).astype("float32").reshape(1, -1)

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               free_text, t_free_text, \
               x_static, y

    def get_item1(self, ind):

        datas = self.get_item(ind)
        if datas is not None:
            x_lab, t_lab, \
            x_vit, t_vit, \
            x_trt, t_trt, \
            free_text, t_free_text, \
            x_static, y = datas
        else:
            return None

        x_lab = torch.from_numpy(x_lab).float()
        t_lab = torch.from_numpy(t_lab).float()

        x_vit = torch.from_numpy(x_vit).float()
        t_vit = torch.from_numpy(t_vit).float()

        x_trt = torch.from_numpy(x_trt).float()
        t_trt = torch.from_numpy(t_trt).float()

        free_text = [torch.from_numpy(free_text[i]).long() for i in range(len(free_text))]

        t_free_text = torch.from_numpy(t_free_text).float()

        x_static = torch.from_numpy(x_static).float()

        y = torch.from_numpy(y).long()

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               free_text, t_free_text, \
               x_static, y

    def get_data(self, inds):

        batches = []
        ids1 = []
        for i in range(len(inds)):
            data = self.get_item1(inds[i])
            if data is None:
                continue
            else:
                batches.append(data)
                ids1.append(inds[i])
        return batches, ids1

    def iterate_batch(self, size, shuffle=True):

        if size > self.len():
            raise ValueError("batch size 大于了总样本数")

        if shuffle:
            all_ids = np.random.choice(self.len(), size=self.len(), replace=False)
        else:
            all_ids = list(range(self.len()))

        if self.len() % size == 0:
            n = self.len() // size
        else:
            n = self.len() // size + 1

        for i in range(n):
            if i == n:
                yield self.get_data(all_ids[(size * i):])
            else:
                yield self.get_data(all_ids[(size * i): (size * (i + 1))])
        return


class EICU_data(object):

    def __init__(self, root, id_file):
        """
        初始化的时候，需要直接输入不同模态的所有数据
        :param root: 包含所有病人文件夹的根文件夹路径

        """
        super().__init__()
        import pandas as pd
        self.root = root
        ds = pd.read_csv(id_file, header=0)
        self.all_stay_id = ds.loc[:, "id"].tolist()
        print('一共有{}人次'.format(self.len()))

    def len(self):
        return (len(self.all_stay_id))

    def get_item(self, ind):
        folder_path = os.path.join(self.root, str(self.all_stay_id[ind]))
        patient = pd.read_csv(os.path.join(folder_path, "patient.csv"), header=0)
        patient = patient.iloc[0, :].tolist()

        # apache = pd.read_csv(os.path.join(folder_path, "apache.csv"), header=0)
        # apache = apache.iloc[0, :].tolist()

        # apache = pd.read_csv(os.path.join(folder_path, "apache.csv"), header=0)
        # apache = apache.iloc[0, :].tolist()

        # x_static = np.array([patient[1:-1] + apache[1:]]).astype("float32").reshape(1, -1)
        x_static = np.array(patient[1:-1]).astype("float32").reshape(1, -1)

        # 诊断标签
        y = np.array([patient[-1]])

        vital_n = pd.read_csv(os.path.join(folder_path, "vital_n.csv"), header=0)

        if vital_n.shape[0] == 0:
            return None

        x_vit = np.asarray(vital_n.iloc[:, 1:]).astype("float32").reshape(1, vital_n.shape[0], -1)
        t_vit = np.asarray(vital_n.iloc[:, 0]).astype("float32").reshape(1, vital_n.shape[0])

        lab_n = pd.read_csv(os.path.join(folder_path, "lab1_n.csv"), header=0)

        if lab_n.shape[0] == 0:
            return None

        x_lab = np.asarray(lab_n.iloc[:, 1:]).astype("float32").reshape(1, lab_n.shape[0], -1)
        t_lab = np.asarray(lab_n.iloc[:, 0]).astype("float32").reshape(1, lab_n.shape[0])

        treatment = pd.read_csv(os.path.join(folder_path, "treatment.csv"), header=0)
        if treatment.shape[0] == 0:
            return None

        x_trt = np.asarray(treatment.iloc[:, 1:]).astype("float32").reshape(1, treatment.shape[0], -1)
        t_trt = np.asarray(treatment.iloc[:, 0]).astype("float32").reshape(1, treatment.shape[0])

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               x_static, y

    def get_item1(self, ind):

        datas = self.get_item(ind)
        if datas is not None:
            x_lab, t_lab, \
            x_vit, t_vit, \
            x_trt, t_trt, \
            x_static, y = datas
        else:
            return None

        x_lab = torch.from_numpy(x_lab).float()
        t_lab = torch.from_numpy(t_lab).float()

        x_vit = torch.from_numpy(x_vit).float()
        t_vit = torch.from_numpy(t_vit).float()

        x_trt = torch.from_numpy(x_trt).float()
        t_trt = torch.from_numpy(t_trt).float()

        x_static = torch.from_numpy(x_static).float()

        y = torch.from_numpy(y).long()

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               x_static, y

    def get_data(self, inds):

        batches = []
        ids1 = []
        for i in range(len(inds)):
            data = self.get_item1(inds[i])
            if data is None:
                continue
            else:
                batches.append(data)
                ids1.append(inds[i])
        return batches, ids1

    def iterate_batch(self, size, shuffle=True):

        if size > self.len():
            raise ValueError("batch size 大于了总样本数")

        if shuffle:
            all_ids = np.random.choice(self.len(), size=self.len(), replace=False)
        else:
            all_ids = list(range(self.len()))

        if self.len() % size == 0:
            n = self.len() // size
        else:
            n = self.len() // size + 1

        for i in range(n):
            if i == n:
                yield self.get_data(all_ids[(size * i):])
            else:
                yield self.get_data(all_ids[(size * i): (size * (i + 1))])
        return


class AP_data(object):

    def __init__(self, root, id_file, wv):
        """
        初始化的时候，需要直接输入不同模态的所有数据
        :param root: 包含所有病人文件夹的根文件夹路径

        """
        super().__init__()
        import pandas as pd
        self.root = root
        ds = pd.read_csv(id_file, header=0)
        self.all_pid = ds.loc[:, "id"].tolist()
        self.w2vmodel = W2Vmodel(wv)
        print('一共有{}人次'.format(self.len()))

    def len(self):
        return (len(self.all_pid))

    def get_item(self, ind):
        folder_path = os.path.join(self.root, str(self.all_pid[ind]))
        state = pd.read_csv(os.path.join(folder_path, "state_info.csv"), header=0)
        x_static = np.array(state).astype("float32").reshape(1, -1)

        # 诊断标签

        label = pd.read_csv(os.path.join(folder_path, "y.csv"), header=0)
        y1 = label.iloc[-1, 1]
        if y1 == 2:
            y = 1
        else:
            y = 0

        y = np.array([y]).reshape(-1, )

        vital_n = pd.read_csv(os.path.join(folder_path, "vit.csv"), header=0)

        if vital_n.shape[0] == 0:
            return None

        x_vit = np.asarray(vital_n.iloc[:, 1:]).astype("float32").reshape(1, vital_n.shape[0], -1)
        t_vit = np.asarray(vital_n.iloc[:, 0]).astype("float32").reshape(1, vital_n.shape[0])

        lab_n = pd.read_csv(os.path.join(folder_path, "lab.csv"), header=0)

        if lab_n.shape[0] == 0:
            return None

        x_lab = np.asarray(lab_n.iloc[:, 1:]).astype("float32").reshape(1, lab_n.shape[0], -1)
        t_lab = np.asarray(lab_n.iloc[:, 0]).astype("float32").reshape(1, lab_n.shape[0])

        treatment = pd.read_csv(os.path.join(folder_path, "drug.csv"), header=0)
        if treatment.shape[0] == 0:
            return None

        x_trt = np.asarray(treatment.iloc[:, 1:]).astype("float32").reshape(1, treatment.shape[0], -1)
        t_trt = np.asarray(treatment.iloc[:, 0]).astype("float32").reshape(1, treatment.shape[0])

        notes = pd.read_csv(os.path.join(folder_path, "note.csv"), header=0)

        free_text = np.array(self.w2vmodel.translate_sentence1(notes.iloc[0, 1].split(" "))).astype("int64").reshape(
            -1, )

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               free_text, \
               x_static, y

    def get_item1(self, ind):
        datas = self.get_item(ind)
        if datas is not None:
            x_lab, t_lab, \
            x_vit, t_vit, \
            x_trt, t_trt, \
            free_text, \
            x_static, y = datas
        else:
            return None

        x_lab = torch.from_numpy(x_lab).float()
        t_lab = torch.from_numpy(t_lab).float()

        x_vit = torch.from_numpy(x_vit).float()
        t_vit = torch.from_numpy(t_vit).float()

        x_trt = torch.from_numpy(x_trt).float()
        t_trt = torch.from_numpy(t_trt).float()

        x_static = torch.from_numpy(x_static).float()

        free_text = torch.from_numpy(free_text).long()

        y = torch.from_numpy(y).long()

        return x_lab, t_lab, \
               x_vit, t_vit, \
               x_trt, t_trt, \
               free_text, \
               x_static, y

    def get_data(self, inds):
        batches = []
        ids1 = []
        for i in range(len(inds)):
            data = self.get_item1(inds[i])
            if data is None:
                continue
            else:
                batches.append(data)
                ids1.append(inds[i])
        return batches, ids1

    def iterate_batch(self, size, shuffle=True):
        if size > self.len():
            raise ValueError("batch size 大于了总样本数")

        if shuffle:
            all_ids = np.random.choice(self.len(), size=self.len(), replace=False)
        else:
            all_ids = list(range(self.len()))

        if self.len() % size == 0:
            n = self.len() // size
        else:
            n = self.len() // size + 1

        for i in range(n):
            if i == n:
                yield self.get_data(all_ids[(size * i):])
            else:
                yield self.get_data(all_ids[(size * i): (size * (i + 1))])
        return


if __name__ == "__main__":
    # data = EICU_data(root="/home/luojiawei/eicu_R_work/all_unitstays/",
    #                  id_file="/home/luojiawei/eicu_R_work/id_files/train_id_discharge_death.csv")
    #
    # x_lab, t_lab, \
    # x_vit, t_vit, \
    # x_trt, t_trt, \
    # x_static, y = data.get_item(1)
    #
    # print(x_lab.shape)

    # w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")
    # print("word2vec load succuess.")
    # data = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
    #                    id_file="/home/luojiawei/mimic3_R_work/id_files/train_id_death24h.csv",
    #                    wv=w2vmodel.wv)
    #
    # x_lab, t_lab, \
    # x_vit, t_vit, \
    # x_trt, t_trt, \
    # free_text, t_free_text, \
    # x_static, y = data.get_item(1)
    #
    # print(x_lab.shape)

    # w2vmodel = Word2Vec.load(
    #     "/home/luojiawei/multimodal_model/data/用于多模态模型的新数据处理 20221015/word2vec_model/word2vec_model.model")
    # print("word2vec load succuess.")
    # data = AP_data(root="/home/luojiawei/RL_AP/data/all_pids/",
    #                id_file="/home/luojiawei/RL_AP/data/id_file.csv",
    #                wv=w2vmodel.wv)
    #
    # x_lab, t_lab, \
    # x_vit, t_vit, \
    # x_trt, t_trt, \
    # free_text, \
    # x_static, y = data.get_item(1)
    #
    # print(x_lab.shape)

    label_name = "death24h"
    checkpoints_path = os.path.join("/tmp/pycharm_project_74", "checkpoints")
    w2vmodel = Word2Vec.load("/home/luojiawei/mimic3_R_work/word2vec_model.model")
    data_te = MIMIC3_data(root="/home/luojiawei/mimic3_R_work/all_admissions/",
                          id_file="/home/luojiawei/mimic3_R_work/id_files/surv_id_" + label_name + ".csv",
                          wv=w2vmodel.wv)

    ids = np.arange(data_te.len())
    batches, _ = data_te.get_data(ids)

    print("hehe")
    pass
