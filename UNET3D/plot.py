import nibabel as nib
from pathlib import Path
import torch
import numpy as np
from torch.nn.functional import one_hot
from skimage.util import montage
import matplotlib.pyplot as plt
import sys
#sys.path.append('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET')
from UNET3D.data_loader import BrainDataset
from unet_model.unet_model import UNet3D
import torch.nn.functional as F
from matplotlib.lines import Line2D

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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[Line2D([0], [0], color='pink', lw=4, label='True'),
                       Line2D([0], [0], color='yellow', lw=4, label='Predicted'),
                       Line2D([0], [0], color='blue', lw=4, label='Intersection')], loc = 'lower center', bbox_to_anchor=(0.5, -0.125), frameon=False)
    return fig

# #Testing data,label,test_label is size 
# data = torch.load('data.pt')
# label = torch.load('label.pt')
# test_label = torch.load('test_label.pt')
# print(f'raw shapes: data: {data.shape}, label: {label.shape}, test_label: {test_label.shape}')

# temp_label = label[:,1:4,:,:,:]
# sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
# mask_true = torch.cat([label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

# temp_label = test_label[:,1:4,:,:,:]
# sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
# mask_pred = torch.cat([test_label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

# image = data.squeeze()[0]

# print(f'plotting shapes: {image.shape}, mask_true: {mask_true.shape}, mask_pred: {mask_pred.shape}')
# #Input shapes are image: [128, 128, 128]), mask_true:[1, 2, 128, 128, 128], mask_pred: [1, 2, 128, 128, 128]
# fig = predictions_plot(image, mask_true, mask_pred)

# plt.show()

# model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device=device)
# print('loading model')
# #Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path(str('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET/data/archive'))
# dataset = BrainDataset(patient_ids, data_dir, binary='TC')
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# print('loading data')
# for batch in train_loader:
#     images, true_masks, patient_ids = batch
#     # mask_true_train = F.one_hot(true_masks[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     mask_pred = model(images.to(device=device, dtype=torch.float32))
#     mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     print(f'mask_pred shape: {mask_pred.shape}')
#     print(f'mask_true shape: {true_masks.shape}')
#     # mask_pred_train = np.argmax(mask_pred.detach().cpu().numpy(), axis=1)
#     # mask_pred_train = F.one_hot(torch.from_numpy(mask_pred_train[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     # img_train = images[0][0]
#     # patient_id = patient_ids[0]
#     # # fig = predictions_plot(img_train, mask_true_train, mask_pred_train, patient_id=patient_id)
#     # # plt.show()
#     break