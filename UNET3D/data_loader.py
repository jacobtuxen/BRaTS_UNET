import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.data = []

        self.init_data()

    def init_data(self):
        ids = ['flair.nii.gz','t1.nii.gz', 't1ce.nii.gz', 't2.nii.gz','seg.nii.gz']
        for patient_id in self.patient_ids:
            self.data.append([nib.load(self.data_dir / patient_id / f'{patient_id}_{id}').get_fdata() for id in ids])
        print(f'len of data: {len(self.data)}')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flair, t1, t1c ,t2, target = self.data[idx]
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

        t1 = t1.unsqueeze(0)
        t1c = t1c.unsqueeze(0)
        t2 = t2.unsqueeze(0)
        flair = flair.unsqueeze(0)
        target = target.unsqueeze(0)

        #Input data size [B,C,W,D,H] 
        #Concatenate t1,t1c,t2,flair
        x = torch.concatenate((t1,t1c,t2,flair), dim=0)
        return x, target
