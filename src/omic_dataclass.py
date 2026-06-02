import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import zscore
import torch
import pickle


class OmicDataset(Dataset):
    def __init__(self, data, nan_mask):
        self.data = data
        self.nan_mask = nan_mask  # True = originally NaN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.nan_mask[idx]


class OmicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        batch_size: int = 8,
        use_log_space: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.use_log_space = use_log_space

    def setup(self, stage=None):
        # 1. Load the data
        with open(self.file_path, "rb") as f:
            matrix_data = pickle.load(f)

        # Convert to tensor (keeping NaNs for now to calculate median properly)
        matrix_tensor = torch.FloatTensor(matrix_data)

        # 2. Perform Splits FIRST to avoid data leakage
        train_idx, temp_idx = train_test_split(
            np.arange(len(matrix_tensor)), test_size=0.3, random_state=42
        )
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Slice raw tensors
        train_m = matrix_tensor[train_idx]
        val_m = matrix_tensor[val_idx]
        test_m = matrix_tensor[test_idx]
        # 3. Record NaN masks before imputation (True = originally NaN)
        train_nan_mask = torch.isnan(train_m)
        val_nan_mask = torch.isnan(val_m)
        test_nan_mask = torch.isnan(test_m)

        # Median Imputation (Strict: Use Train Medians for all sets)
        train_medians = torch.nanmedian(train_m, dim=0, keepdim=True).values
        train_medians = torch.nan_to_num(train_medians, nan=0.0)

        def fill_nan(t, medians):
            is_nan = torch.isnan(t)
            t[is_nan] = medians.expand_as(t)[is_nan]
            return t

        train_m = fill_nan(train_m, train_medians)
        val_m = fill_nan(val_m, train_medians)
        test_m = fill_nan(test_m, train_medians)

        # 4. Winsorize per feature: clamp to 1st/99th percentiles (computed from train only)
        train_np = train_m.numpy()
        p1 = np.percentile(train_np, 1, axis=0, keepdims=True)
        p99 = np.percentile(train_np, 99, axis=0, keepdims=True)
        p1_t = torch.from_numpy(p1)
        p99_t = torch.from_numpy(p99)

        train_m = train_m.clamp(min=p1_t, max=p99_t)
        val_m = val_m.clamp(min=p1_t, max=p99_t)
        test_m = test_m.clamp(min=p1_t, max=p99_t)

        # 5. Z-score normalization via scipy (fit on train, apply to all)
        train_np = train_m.numpy()
        train_mean = np.mean(train_np, axis=0, keepdims=True)
        train_std = np.std(train_np, axis=0, keepdims=True)
        # zscore with pre-computed train stats: (x - train_mean) / train_std
        # Use nan_policy='omit' not needed since NaNs are already imputed
        train_m = torch.from_numpy(zscore(train_np, axis=0, nan_policy="omit")).float()
        val_m = torch.from_numpy(
            ((val_m.numpy() - train_mean) / (train_std + 1e-8))
        ).float()
        test_m = torch.from_numpy(
            ((test_m.numpy() - train_mean) / (train_std + 1e-8))
        ).float()

        train_m = fill_nan(train_m, torch.zeros_like(train_medians))
        val_m = fill_nan(val_m, torch.zeros_like(train_medians))
        test_m = fill_nan(test_m, torch.zeros_like(train_medians))

        # 6. Optional log transform: log1p handles zeros safely
        if self.use_log_space:
            train_m = torch.log1p(train_m.clamp(min=0))
            val_m = torch.log1p(val_m.clamp(min=0))
            test_m = torch.log1p(test_m.clamp(min=0))

        assert not torch.any(torch.isnan(train_m)), "NANs in train DATA!"
        assert not torch.any(torch.isnan(val_m)), "NANs in val DATA!"
        assert not torch.any(torch.isnan(test_m)), "NANs in test DATA!"

        # 7. Assign to Dataset objects
        self.train_ds = OmicDataset(train_m, train_nan_mask)
        self.val_ds = OmicDataset(val_m, val_nan_mask)
        self.test_ds = OmicDataset(test_m, test_nan_mask)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
        )
