import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path, binary='WT'): #(WT, TC, MT)
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.binary = binary
        self.extensions = ['flair.nii', 't1ce.nii', 't2.nii','seg.nii']

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_paths = [self.data_dir / patient_id / f'{patient_id}_{data_id}' for data_id in self.extensions]
        data = [self.load_nifti_file(path) for path in data_paths]
        target = torch.from_numpy(np.where(data[-1]==4, 3, data[-1])).long()
        if self.binary == 'WT':
            target = torch.where(target==0, 0, 1)
        elif self.binary == 'MT':
            target = torch.where(target==3, 1, target)
            target = torch.where(target==2, 0, target)
        elif self.binary == 'TC':
            target = torch.where(target==1, 1, 0)
        else:
            raise ValueError('binary must be one of: WT, MT, TC')
        data = torch.cat([torch.from_numpy(data[i]).unsqueeze(0) for i in range(len(self.extensions)-1)], dim=0)

        start_idx = 56
        end_idx = 184
        start_idx_height = 13
        end_idx_height = 141
        
        data = data[:,start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]
        target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

        #normalize data in each channel min max normalization
        for i in range(data.shape[0]):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

        return data, target, patient_id

# #Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET' / 'data' / 'archive'
# dataset = BrainDataset(patient_ids, data_dir, binary='TC')
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)
# #visualize target
# import matplotlib.pyplot as plt
# plt.imshow(target[:,:,70])
# plt.show()