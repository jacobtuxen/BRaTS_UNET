import nibabel as nib
import numpy as np
from pathlib import Path
import torch
from torchmetrics import JaccardIndex, ConfusionMatrix, Dice

#For HPC
#data_dir = Path('/work3/s211469/data')
#val_ids = np.loadtxt(data_dir / 'filenames_val.txt', dtype=str)
#true_masks = [nib.load(data_dir / f'{patient}seg.nii').get_fdata() for patient in val_ids]

#For local
data_dir = Path(str('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET/data/archive/BraTS2021_00495'))
val_ids = ['BraTS2021_00495']
true_masks = [nib.load(data_dir / f'{patient}seg.nii').get_fdata() for patient in val_ids]

models = ['model_1.pt', 'model_2.pt', 'model_3.pt']

predictions = [np.sum([model(val_patient).argmax(dim=1).cpu().numpy() for model in models],axis=0) for val_patient in val_ids]

num_classes = 4
jaccard = JaccardIndex(task="multiclass" ,num_classes=num_classes)
dice = Dice(num_classes=num_classes)
confusionmat = ConfusionMatrix(task = 'multiclass', num_classes=num_classes)

jaccard_scores = [jaccard(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]
dice_scores = [dice(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]
confusion = [confusionmat(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]

# #For testing
# # Make 3 predictions of flots size [batch,2,128,128]
# batch_size = 4
# output1 = torch.rand(batch_size,2,128,128,128)
# output2 = torch.rand(batch_size,2,128,128,128)
# output3 = torch.rand(batch_size,2,128,128,128)

# pred1 = output1.argmax(dim=1).cpu().numpy()
# pred2 = output2.argmax(dim=1).cpu().numpy()
# pred3 = output3.argmax(dim=1).cpu().numpy()

# print(f'pred1 shape: {pred1.shape}, {np.unique(pred1)}')
# print(f'pred2 shape: {pred2.shape}, {np.unique(pred2)}')
# print(f'pred3 shape: {pred3.shape}, {np.unique(pred3)}')

# total_predictions = np.sum([pred1, pred2, pred3], axis=0)
# print(f'total_predictions shape: {total_predictions.shape}, {np.unique(total_predictions)}')
