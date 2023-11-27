from pathlib import Path
import nibabel as nib
import numpy as np

# data_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET' / 'data' / 'archive'
data_dir = Path('/work3/s211469/data')
patient_ids = np.loadtxt(data_dir / 'filenames.txt', dtype=str)


end_patients = []
threshold = 200000

for patient_id in patient_ids:
  #count classes and all voxels
  seg = nib.load(data_dir / patient_id / f'{patient_id}_seg.nii').get_fdata()
  sum_seg = np.sum(seg)
  if sum_seg > threshold:
    end_patients.append(patient_id)
    print('Patient added')
np.savetxt(data_dir / 'filenames_filtered.txt', end_patients, fmt='%s')