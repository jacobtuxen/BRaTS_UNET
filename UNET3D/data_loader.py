import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import pywt

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path, binary='WT', reconstruction_wavelet = 'haar', threshold = 0): #(WT, TC, MT)
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.binary = binary
        self.extensions = ['flair.nii', 't1ce.nii', 't2.nii','seg.nii']
        self.wavelet = reconstruction_wavelet
        self.keys = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        self.threshold_percentile = threshold

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def wavelet_reconstruction(self, data):
        #make wavelet transform, then threshold, then inverse wavelet transform
        #data is a 4D tensor
        data = data.numpy()
        reconstructed_data = []
        for i in range(data.shape[0]):
            coeffs_of_levels = pywt.wavedecn(data[i], self.wavelet, level=4)
            temp = []
            for idx, coeffs in enumerate(coeffs_of_levels):
                for idxj, key in enumerate(self.keys):
                    if idx != 0:
                        temp.append(coeffs[key].reshape((1,-1)))
                    else:
                        temp.append(coeffs[0].reshape((1,-1)))
            temp = np.concatenate(temp, axis=1)
            threshold = np.percentile(np.abs(temp), self.threshold_percentile)
            count_zeroes = 0
            for idx, coeffs in enumerate(coeffs_of_levels):
                for idxj, key in enumerate(self.keys):
                    if idx != 0:
                        coeffs[key] = np.where(np.abs(coeffs[key]) < threshold, 0, coeffs[key])
                        count_zeroes += np.where(coeffs[key]==0, 1, 0).sum()
                    else:
                        coeffs[0] = np.where(np.abs(coeffs[0]) < threshold, 0, coeffs[0])
                        count_zeroes += np.where(coeffs[0]==0, 1, 0).sum()
            reconstructed_data.append(pywt.waverecn(coeffs_of_levels, self.wavelet))
            print(f'Percentage of zeroes: {count_zeroes/temp.shape[1]}')
        reconstructed_data = np.stack(reconstructed_data, axis=0)

        return torch.from_numpy(reconstructed_data), coeffs_of_levels


    
    
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
        data, coeffs_level = self.wavelet_reconstruction(data)
        target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

        #normalize data in each channel min max normalization
        for i in range(data.shape[0]):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

        return data, target, patient_id, coeffs_level

#Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET' / 'data' / 'archive'
# dataset = BrainDataset(patient_ids, data_dir, binary='TC', threshold=0)
# data, target,_, coeffs_level = dataset[0]
# print(data.shape)
# print(target.shape)
# #plot
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(4, 4)
# axs[0][0].imshow(data[0,:,:,70], cmap='gray')
# axs[0][1].imshow(data[1,:,:,70], cmap='gray')
# axs[0][2].imshow(data[2,:,:,70], cmap='gray')
# axs[0][3].imshow(target[:,:,70], cmap='gray')
# axs[1][0].imshow(coeffs_level[4]['aad'][:,:,35], cmap='gray')
# axs[1][1].imshow(coeffs_level[4]['ada'][:,:,35], cmap='gray')
# axs[1][2].imshow(coeffs_level[4]['add'][:,:,35], cmap='gray')
# axs[1][3].imshow(coeffs_level[4]['daa'][:,:,35], cmap='gray')
# axs[2][0].imshow(coeffs_level[4]['dad'][:,:,35], cmap='gray')
# axs[2][1].imshow(coeffs_level[4]['dda'][:,:,35], cmap='gray')
# axs[2][2].imshow(coeffs_level[4]['ddd'][:,:,35], cmap='gray')
# axs[2][3].imshow(coeffs_level[3]['aad'][:,:,35//2], cmap='gray')
# axs[3][0].imshow(coeffs_level[3]['ada'][:,:,35//2], cmap='gray')
# axs[3][1].imshow(coeffs_level[3]['add'][:,:,35//2], cmap='gray')
# axs[3][2].imshow(coeffs_level[3]['daa'][:,:,35//2], cmap='gray')
# axs[3][3].imshow(coeffs_level[3]['dad'][:,:,35//2], cmap='gray')
# plt.show()
