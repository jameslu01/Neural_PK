import utils
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TDM1(Dataset):
    """A class that processes data of the type used in this paper.

    From the paper: "This study utilized clinical PK data from
    trastuzumab emtansine (T-DM1), a conjugated monoclonal antibody
    drug that has been approved for the treatment of patients
    with human epidermal growth factor receptor 2 (HER2)
    positive breast cancers (Boyraz et al., 2013).
    """

    def __init__(self, data, label_col, feature_cols):
        self.data = data

        self.label_col = label_col
        self.features = feature_cols
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

        if (data.DSFQ == 1).all():  # 1 week
            cmax_time = data.loc[data.TIME < 7, ["TIME", "PK_timeCourse"]].values.flatten()
            cmax = data.loc[data.TIME < 7, "PK_timeCourse"].max()
        else:  # 3 weeks
            cmax_time = data.loc[data.TIME < 21, ["TIME", "PK_timeCourse"]].values.flatten()
            cmax = data.loc[data.TIME < 21, "PK_timeCourse"].max()

        # from the paper:
        # "We iterated TIME and PK values from the first time point
        # to the last time point during the first cycle, and padded
        # the vector to 20 elements long with zeros"
        cmax_time_full = np.zeros((20,))
        if len(cmax_time) <= 20:
            cmax_time_full[: len(cmax_time)] = cmax_time
        else:
            cmax_time_full[:] = cmax_time[:20]

        data.loc[:, "PK_round1"] = data["PK_round1"] / cmax

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

    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
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

        indices = inverse_indices[offset : offset + len(tt)]
        offset += len(tt)

        combined_features[b, indices] = features.float()
        combined_label[b, indices] = label.float()
        combined_cmax_time[b, :] = cmax_time.float()
    combined_tt = combined_tt.float()

    return ptnms, combined_tt, combined_features, combined_label, combined_cmax_time


def parse_tdm1(device, train, validate, test, phase="train"):
    """This function constructs the various"""
    feature_cols = ["TFDS", "TIME", "CYCL", "AMT", "PK_round1"]
    label_col = "PK_timeCourse"
    if phase == "train":
        train = TDM1(train, label_col, feature_cols)
        validate = TDM1(validate, label_col, feature_cols)
        ptnm, times, features, labels, cmax_time = train[0]

        train_dataloader = DataLoader(
            train, batch_size=1, shuffle=True, collate_fn=lambda batch: tdm1_collate_fn(batch, device)
        )
        val_dataloader = DataLoader(
            validate, batch_size=1, shuffle=False, collate_fn=lambda batch: tdm1_collate_fn(batch, device)
        )

        dataset_objs = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "input_dim": features.size(-1),
        }

    else:
        test = TDM1(test, label_col, feature_cols)
        ptnm, times, features, labels, cmax_time = test[0]
        test_dataloader = DataLoader(
            test, batch_size=1, shuffle=False, collate_fn=lambda batch: tdm1_collate_fn(batch, device)
        )

        dataset_objs = {
            "test_dataloader": utils.inf_generator(test_dataloader),
            "n_test_batches": len(test_dataloader),
            "input_dim": features.size(-1),
        }

    return dataset_objs
