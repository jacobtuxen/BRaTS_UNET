from pathlib import Path
import nibabel as nib
import numpy as np

# data_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET' / 'data' / 'archive'
data_dir = Path('/work3/s194572/data')
patient_ids = np.loadtxt(data_dir / 'filenames_filtered.txt', dtype=str)

all_classes = np.zeros(4)
for patient_id in patient_ids:
  #count classes and all voxels
  seg = nib.load(data_dir / patient_id / f'{patient_id}_seg.nii').get_fdata()
  classes, counts = np.unique(seg, return_counts=True)
  for c, count in zip(classes, counts):
    if c == 4:
      c = 3
    all_classes[int(c)] += count
sum_all_classes = np.sum(all_classes)
print(sum_all_classes/all_classes)