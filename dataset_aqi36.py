import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
from utils import get_randmask, get_hist_mask


class AQI36_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", val_len=0.1, is_interpolate=False,
                 target_strategy='hybrid', mask_sensor=None, missing_ratio=None):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor

        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 1st,4th,7th,10th months are excluded from histmask (since the months are used for creating missing patterns in test dataset)
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]
        elif mode == "valid":
            month_list = [2, 5, 8, 11]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        self.month_list = month_list

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        df = pd.read_csv(
            "./data/pm25/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        df_gt = pd.read_csv(
            "./data/pm25/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )

        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]]
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]
            if mode == 'train' and month_list[i] in [2, 5, 8, 11]:
                cut_len = int(val_len * len(current_df))
                current_df = current_df[:-cut_len]
                current_df_gt = current_df_gt[:-cut_len]
            if mode == 'valid':
                cut_len = int(val_len * len(current_df))
                current_df = current_df[-cut_len:]
                current_df_gt = current_df_gt[-cut_len:]
            current_length = len(current_df) - eval_length + 1

            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask
            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            if mode == "test":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month


    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]

        index2 = np.random.randint(0, len(self.use_index))
        hist_month = self.index_month_histmask[index2]
        hist_index = self.position_in_month_histmask[index2]

        ob_data = self.observed_data[c_month][c_index:c_index + self.eval_length]
        ob_mask = self.observed_mask[c_month][c_index:c_index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[c_month][c_index:c_index + self.eval_length]
        for_pattern_mask = self.observed_mask[hist_month][hist_index:hist_index + self.eval_length]

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_hist_mask(ob_mask_t, for_pattern_mask=for_pattern_mask)
            else:
                cond_mask = get_randmask(ob_mask_t)

        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "hist_mask": for_pattern_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask.numpy()
        }
        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size, device, val_len=0.1, is_interpolate=False, num_workers=4, target_strategy='hybrid'):
    dataset = AQI36_Dataset(mode="train", is_interpolate=is_interpolate, target_strategy=target_strategy)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = AQI36_Dataset(mode="test", is_interpolate=is_interpolate, target_strategy=target_strategy)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = AQI36_Dataset(mode="valid", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler

