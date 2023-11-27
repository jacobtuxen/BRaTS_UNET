import nibabel as nib
from pathlib import Path
import torch
import numpy as np
from torch.nn.functional import one_hot
from skimage.util import montage
import matplotlib.pyplot as plt

def predictions_plot(image, mask_true, mask_pred, patient_id='BraTS2021_00495'):
    #Args:
    #mask_true: true binary mask is a tensor of size (1, 2, 128,128,128)
    #mask_pred: predicted binary mask is a tensor of size (1, 2, 128,128,128)
    #image: is a flair data tensor of size (128,128,128)
    #patient_id: is a string of the patient id
    assert len(mask_true.shape) == 5
    assert len(mask_pred.shape) == 5
    assert len(image.shape) == 3

    img_tensor = image.detach().numpy()
    mask_true_tensor = mask_true.squeeze()[1].squeeze().cpu().detach().numpy()
    mask_pred_tensor = mask_pred.squeeze()[1].squeeze().cpu().detach().numpy()

    #Pad zeros to size 160x160x160
    img_tensor = np.pad(img_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_true_tensor = np.pad(mask_true_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_pred_tensor = np.pad(mask_pred_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)


    image = np.rot90(montage(img_tensor))
    mask_true = np.rot90(montage(mask_true_tensor))
    mask_pred = np.rot90(montage(mask_pred_tensor))


    intersection = np.logical_and(mask_true, mask_pred)
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    plt.subplot(1,1,1)
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask_true == False, mask_true),
                cmap='cool', alpha=0.9)
    ax.imshow(np.ma.masked_where(mask_pred == False, mask_pred),
                cmap='spring', alpha=0.6)
    ax.imshow(np.ma.masked_where(intersection == False, intersection),
                cmap='winter', alpha=1)
    plt.suptitle(f'Error plot patient: {patient_id}')
    return fig

def predictions_plot(image, mask_true, mask_pred, patient_id='BraTS2021_00495'):
    #Args:
    #mask_true: true binary mask is a tensor of size (1, 2, 128,128,128)
    #mask_pred: predicted binary mask is a tensor of size (1, 2, 128,128,128)
    #image: is a flair data tensor of size (128,128,128)
    #patient_id: is a string of the patient id
    assert len(mask_true.shape) == 5
    assert len(mask_pred.shape) == 5
    assert len(image.shape) == 3

    img_tensor = image.detach().numpy()
    mask_true_tensor = mask_true.squeeze()[1].squeeze().cpu().detach().numpy()
    mask_pred_tensor = mask_pred.squeeze()[1].squeeze().cpu().detach().numpy()

    #Pad zeros to size 160x160x160
    img_tensor = np.pad(img_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_true_tensor = np.pad(mask_true_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_pred_tensor = np.pad(mask_pred_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    

    image = np.rot90(montage(img_tensor))
    mask_true = np.rot90(montage(mask_true_tensor))
    mask_pred = np.rot90(montage(mask_pred_tensor))


    intersection = np.logical_and(mask_true, mask_pred)
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    plt.subplot(1,1,1)
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask_true == False, mask_true),
                cmap='cool', alpha=0.9)
    ax.imshow(np.ma.masked_where(mask_pred == False, mask_pred),
                cmap='spring', alpha=0.6)
    ax.imshow(np.ma.masked_where(intersection == False, intersection),
                cmap='winter', alpha=1)
    plt.suptitle(f'Error plot patient: {patient_id}')
    return fig

#Testing data,label,test_label is size 
data = torch.load('data.pt')
label = torch.load('label.pt')
test_label = torch.load('test_label.pt')
print(f'raw shapes: data: {data.shape}, label: {label.shape}, test_label: {test_label.shape}')

temp_label = label[:,1:4,:,:,:]
sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
mask_true = torch.cat([label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

temp_label = test_label[:,1:4,:,:,:]
sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
mask_pred = torch.cat([test_label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

image = data.squeeze()[0]

print(f'plotting shapes: {image.shape}, mask_true: {mask_true.shape}, mask_pred: {mask_pred.shape}')
#Input shapes are image: [128, 128, 128]), mask_true:[1, 2, 128, 128, 128], mask_pred: [1, 2, 128, 128, 128]
fig = predictions_plot(image, mask_true, mask_pred)

plt.show()