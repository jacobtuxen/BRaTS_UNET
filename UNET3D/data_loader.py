import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path):
        self.patient_ids = patient_ids
        self.data_dir = data_dir

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_paths = [self.data_dir / patient_id / f'{patient_id}_{data_id}' for data_id in ['flair.nii.gz','t1.nii.gz', 't1ce.nii.gz', 't2.nii.gz','seg.nii.gz']]
        data = [self.load_nifti_file(path) for path in data_paths]
        target = torch.from_numpy(np.where(data[4]==4, 3, data[4])).long()
        #Cat
        data = torch.cat([torch.from_numpy(data[i]).unsqueeze(0) for i in range(4)], dim=0)

        start_idx = (data.shape[1]-160)//2
        end_idx = (data.shape[1]+160)//2
        
        data = data[:,start_idx:end_idx,start_idx:end_idx,:]
        target = target[start_idx:end_idx,start_idx:end_idx,:]
        
        #Normalize
        data = F.normalize(data, p=2, dim=0)
        

        return data, target

#Test loader    
patient_ids = ['BraTS2021_00495']
data_dir = Path.home() / 'Documents' / 'DTU' / 'E23' / '02456_Deep_Learning' / 'Brain_Project' / 'BRaTS_UNET' / 'data' / 'archive'
dataset = BrainDataset(patient_ids, data_dir)
data, target = dataset[0]
print(data.shape)
print(target.shape)