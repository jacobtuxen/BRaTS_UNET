import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.transform = False
        self.add_noise = False

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_paths = [self.data_dir / patient_id / f'{patient_id}_{data_id}' for data_id in ['flair.nii','t1.nii', 't1ce.nii', 't2.nii','seg.nii']]
        data = [self.load_nifti_file(path) for path in data_paths]
        target = torch.from_numpy(np.where(data[4]==4, 3, data[4])).long()
        #Cat
        data = torch.cat([torch.from_numpy(data[i]).unsqueeze(0) for i in range(4)], dim=0)

        start_idx = (data.shape[1]-160)//2
        end_idx = (data.shape[1]+160)//2

        data = F.pad(data, (0, 160 - data.shape[3], 0, 0)) 
        target = F.pad(target, (0, 160 - target.shape[2], 0, 0, 0, 0))
        
        data = data[:,start_idx:end_idx,start_idx:end_idx,:]
        target = target[start_idx:end_idx,start_idx:end_idx,:]
        
        #Normalize
        data = F.normalize(data, p=2, dim=0)
        
        if self.transform:
            data = self.data_transform(data)

        if self.add_noise:
            data = self.add_weight_noise(data)
        
        return data, target, patient_id
    
    def data_transform(self, data):
        #rotate 3d input image
        data = torch.rot90(data, k=np.random.randint(1,4), dims=(1,2))
        #flip 3d input image
        if np.random.randint(1,4):
            data = torch.flip(data, dims=(1,2))
        return data
    
    #add random noise to data input
    def add_weight_noise(self, data):
        sigma = 25.0
        noise = sigma * torch.randn_like(data)
        return data + noise

#Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path.home() / 'Documents' / 'DTU' / 'E23' / '02456_Deep_Learning' / 'Brain_Project' / 'BRaTS_UNET' / 'data' / 'archive'
# dataset = BrainDataset(patient_ids, data_dir)
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)