import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

extensions = ['flair.nii.gz', 't1.nii.gz', 't1ce.nii.gz', 't2.nii.gz', 'seg.nii.gz']
modality_names = ['FLAIR', 'T1', 'T1CE', 'T2', 'Segmentation']
path = Path(str('/Users/madssverker/Documents/GitHub/BRaTS_UNET/data/archive/BraTS2021_00495'))
patient_id = 'BraTS2021_00495'
data = [nib.load(path / f'{patient_id}_{ext}').get_fdata() for ext in extensions]
slice_idx = 100
#5x1 subplot horizontal
fig, ax = plt.subplots(1, 5, figsize=(15, 5))
for i in range(len(extensions)):
        if i == len(extensions)-1:
            ax[i].imshow(data[i][:, :, slice_idx], cmap='gnuplot')
            ax[i].legend(handles=[Line2D([0], [0], color='yellow', lw=4, label='Necrotic and Non-Enhancing Tumor'),
                                  Line2D([0], [0], color='red', lw=4, label='Edema'),
                                  Line2D([0], [0], color='purple', lw=4, label='Enhancing Tumor'),
                                  Line2D([0], [0], color='black', lw=4, label='Background')],
                         loc='lower center', fontsize='7', bbox_to_anchor=(0.53, -0.4), frameon=False)
        else:
            img = ax[i].imshow(data[i][:, :, slice_idx], cmap='bone')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f'{modality_names[i]}')

# Adjust layout to make room for the colorbar at the bottom
fig.subplots_adjust(bottom=0.2)

# Position the colorbar below the subplots
cbar_ax = fig.add_axes([0.125, 0.25, 0.615, 0.05])  # Adjust these values as needed
cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cbar.set_ticklabels(['0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875'])
plt.show()


