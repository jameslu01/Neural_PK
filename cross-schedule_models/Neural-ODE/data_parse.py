import utils
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

class TDM1(Dataset):

    def __init__(self, data_to_load, label_col, feature_cols, device, phase="train"):
        self.data = pd.read_csv(data_to_load)

        self.label_col = label_col
        self.features = feature_cols
        self.device = device
        self.phase = phase
        self.data["TIME"] = self.data["TIME"] / 24

    def __len__(self):
        return self.data.PTNM.unique().shape[0] 

    def __getitem__(self, index):
        ptnm = self.data.PTNM.unique()[index]
        cur_data = self.data[self.data["PTNM"] == ptnm]
        times, features, labels, cmax_time = self.process(cur_data)
        return ptnm, times, features, labels, cmax_time

    def process(self, data):
        data = data.reset_index(drop=True)
        """
        if self.phase == "train":
            random_time = np.random.randint(low=21, high=int(data["TIME"].max()) + 1)
            data = data[data.TIME <= random_time]
        else:
            pass
        """
        if (data.DSFQ == 1).all():
            cmax_time = data.loc[data.TIME < 7, ["TIME", "PK_timeCourse"]].values.flatten()
            cmax = data.loc[data.TIME < 7, "PK_timeCourse"].max()
        else:
            cmax_time = data.loc[data.TIME < 21, ["TIME", "PK_timeCourse"]].values.flatten()
            cmax = data.loc[data.TIME < 21, "PK_timeCourse"].max()

        cmax_time_full = np.zeros((20, ))
        if len(cmax_time) <= 20:
            cmax_time_full[:len(cmax_time)] = cmax_time
        else:
            cmax_time_full[:] = cmax_time[:20]
        
        data["PK_round1"] = data["PK_round1"] / cmax

        features = data[self.features].values
        labels = data[self.label_col].values
        times = data["TIME"].values

        times = torch.from_numpy(times)
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels).unsqueeze_(-1)
        cmax_time_full = torch.from_numpy(cmax_time_full)
        return times, features, labels, cmax_time_full


def tdm1_collate_fn(batch, device):
    D = batch[0][2].shape[1]
    N = 1

    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]),
        sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)
    # print(combined_tt, inverse_indices)

    offset = 0
    combined_features = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_label = torch.zeros([len(batch), len(combined_tt), N]).to(device)
    combined_label[:] = np.nan
    combined_cmax_time = torch.zeros([len(batch), 20]).to(device)
    # print(combined_label.shape)

    ptnms = []
    for b, (ptnm, tt, features, label, cmax_time) in enumerate(batch):
        ptnms.append(ptnm)
        tt = tt.to(device)
        features = features.to(device)
        label = label.to(device)
        cmax_time = cmax_time.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_features[b, indices] = features.float()
        combined_label[b, indices] = label.float()
        combined_cmax_time[b, :] = cmax_time.float()
    combined_tt = combined_tt.float()

    return ptnms, combined_tt, combined_features, combined_label, combined_cmax_time


def parse_tdm1(device, phase="train"):
    train_data_path = "train.csv"
    val_data_path = "validate.csv"
    test_data_path = "test.csv"
    test_interp_path = "test_interp.csv"
    test_nodosing_path = "test_nodosing.csv"

    feature_cols = ['TFDS','TIME','CYCL','AMT',"PK_round1"]
    """
    covariates = ['SEX','AGE','WT','RACR','RACE','BSA',
                  'BMI','ALBU','TPRO','WBC','CRCL','CRET',
                  'SGOT','SGPT','TBIL','TMBD','ALKP', 'HER', 
                  'ECOG','KEOALL','ASIAN']
    feature_cols += covariates
    """
    label_col = "PK_timeCourse"
    train = TDM1(train_data_path, label_col, feature_cols, device, phase="train")
    validate = TDM1(val_data_path, label_col, feature_cols, device, phase="validate")
    test = TDM1(test_data_path, label_col, feature_cols, device, phase="test")
    test_interp = TDM1(test_interp_path, label_col, feature_cols, device, phase="test")
    test_nodosing = TDM1(test_nodosing_path, label_col, feature_cols, device, phase="test")

    ptnm, times, features, labels, cmax_time = train[0]
    input_dim = features.size(-1)
    # n_labels = 1

    if phase == "train":
        train_dataloader = DataLoader(train, batch_size=1, shuffle=True, 
            collate_fn=lambda batch: tdm1_collate_fn(batch, device))
        val_dataloader = DataLoader(validate, batch_size=1, shuffle=False,
            collate_fn=lambda batch: tdm1_collate_fn(batch, device))

        dataset_objs = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "input_dim": input_dim
        }

    else:
        test_dataloader = DataLoader(test, batch_size=1, shuffle=False,
            collate_fn=lambda batch: tdm1_collate_fn(batch, device))
        interp_dataloader = DataLoader(test_interp, batch_size=1, shuffle=False,
            collate_fn=lambda batch: tdm1_collate_fn(batch, device))
        nodosing_dataloader = DataLoader(test_nodosing, batch_size=1, shuffle=False,
            collate_fn=lambda batch: tdm1_collate_fn(batch, device))

        dataset_objs = {
            "test_dataloader": utils.inf_generator(test_dataloader),
            "interp": utils.inf_generator(interp_dataloader),
            "nodosing": utils.inf_generator(nodosing_dataloader),
            "n_test_batches": len(test_dataloader),
            "input_dim": input_dim
        }

    return dataset_objs

    




if __name__ == "__main__":
    print("run")
    data = parse_tdm1("cpu")
    for ptnms, times, features, labels, cmax_time in data["train_dataloader"]:
        print(ptnms)
        print(times)
        print(features)
        print(labels)
        print(cmax_time)
        break

