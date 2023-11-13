import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_model_output(epoch, input, model, patient_id, device):
  slices = np.linspace(10,150, 5)
  plot_titles = ['flair','t1', 't1ce', 't2','seg']

  model.eval()
  with torch.no_grad():
    input = input.squeeze(0).to(device)
    output = model(input)
  
  fig, ax = plt.subplots(len(slices), 6, figsize=(15, 5))
  originals = [nib.load(f'/work3/s211469/data/{patient_id}/{patient_id}_{titles}').get_fdata() for titles in ['flair.nii','t1.nii', 't1ce.nii', 't2.nii','seg.nii']]
  for idj, slice_ in enumerate(slices):
    for idx, original in enumerate(originals):
      ax[idj, idx].imshow(original[:,:,slice_], cmap='gray')
      ax[idj, idx].axis('off')
      if idj == 0 and idx == 0:
        ax[0, idx].set_title(f'E: {epoch}, {plot_titles[idx]}')
      elif idj == 0:
        ax[0, idx].set_title(f'{plot_titles[idx]}')
    ax[idj, 5].imshow(output[:,:,slice_], cmap='gray')
    ax[idj, 5].axis('off')
    if idj == 0:
      ax[0, 5].set_title(f'Prediction')
  model.train()
  return fig