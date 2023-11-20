import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_model_output(epoch, input, model, patient_id, device):
  slices = np.linspace(10,100, 5)
  plot_titles = ['flair', 't1ce', 't2','seg', 'output']

  model.eval()
  with torch.no_grad():
    input = input.unsqueeze(0).to(device)
    output = model(input).cpu().numpy()
    output = np.argmax(output, axis=1)
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