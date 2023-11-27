import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.util import montage

def visualize_model_output(epoch, input, model, patient_id, device):
  slices = np.linspace(10,100, 5)
  plot_titles = ['flair', 't1ce', 't2','seg', 'output']

  model.eval()
  with torch.no_grad():
    input = input.unsqueeze(0).to(device)
    output = model(input).cpu().numpy()
    output = np.argmax(output, axis=1) #<- #0 air
    input_images = input.cpu().numpy()

  input_images = np.squeeze(input_images, axis=0)
  fig, ax = plt.subplots(len(slices), 5, figsize=(15, 5))
  seg = [nib.load(f'/work3/s194572/data/{patient_id}/{patient_id}_{titles}').get_fdata() for titles in ['seg.nii']]

  for i, slice in enumerate(slices):
    for j in range(len(plot_titles)):
      if j == 0:
        ax[i, j].imshow(input_images[0, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 1:
        ax[i, j].imshow(input_images[1, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 2:
        ax[i, j].imshow(input_images[2, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 3:
        ax[i, j].imshow(seg[0][:, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 4:
        ax[i, j].imshow(output[0, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      if i==0:
        ax[i, j].set_title(f'{plot_titles[j]}')
      
  model.train()
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

