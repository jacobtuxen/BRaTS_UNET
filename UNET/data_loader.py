import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path

class BrainDataset(Dataset):
    def __init__(self, filenames: list, data_dir: Path):
        self.filenames = filenames
        self.data_dir = data_dir
        self.data = []

        self.init_data()

    def init_data(self):
        ids = ['flair.nii.gz','t1.nii.gz', 't1ce.nii.gz', 't2.nii.gz','seg.nii.gz']
        for id in ids:
            self.data.append([nib.load(self.data_dir / filename / f'{filename}_{id}').get_fdata() for filename in self.filenames])

    def __len__(self):
        return len(self.data[0][0,0,:])

    def __getitem__(self, idx):
        t1, t1c ,t2, flair, target = [self.data[i][:,:,idx] for i in range(5)]
        t1 = torch.from_numpy(t1).float()
        t1c = torch.from_numpy(t1c).float()
        t2 = torch.from_numpy(t2).float()
        flair = torch.from_numpy(flair).float()
        target = torch.from_numpy(target).float()

        #Normalize
        t1 = (t1 - t1.mean()) / t1.std()+1e-8
        t1c = (t1c - t1c.mean()) / t1c.std()+1e-8
        t2 = (t2 - t2.mean()) / t2.std()+1e-8
        flair = (flair - flair.mean()) / flair.std()+1e-8
        return t1, t1c ,t2, flair, target
