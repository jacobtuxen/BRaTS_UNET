import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

data_dir = Path('/work3/s211469/data')
patient_ids = np.loadtxt(data_dir / 'filenames.txt', dtype=str)
extension = 'seg.nii'
distributions = []

for patient_id in patient_ids:
    data_path = data_dir / patient_id / f'{patient_id}_{extension}'
    target = nib.load(data_path).get_fdata()
    
    start_idx = 56
    end_idx = 184
    start_idx_height = 13
    end_idx_height = 141
    target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

    target = np.where(target==4, 3, target)

    distribution = np.bincount(target.flatten().astype(int))
    distributions.append(distribution)
    print(f'Patient {patient_id} done')
#Save distributions
distributions = np.asarray(distributions)
np.save('class_distributions.npy', distributions)