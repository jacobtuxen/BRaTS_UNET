# BRaTS_UNET
3dUNET for BRaTS 2021 dataset

This code is written for DTU's HPC. To run this code, the user must have placed the BraTS data set in the scratch directory of the user on HPC. However, adaptations are very much possible. Please take a look at the guide to running code.

Step-by-step guide to run the code:
1) data_dir in output_merged_file.py to the directory where the user has BraTS 2021 data.
2) Change WandB API key to your own in output_merged_file.py. Or optionally set USE_WANDB to False in output_merged_file.py.
3) Create a filenames_filtered.txt in your data_dir folder; this .txt should have the names of the patients you want to train/validate on. An example is uploaded on this GitHub.
4) Train the model.
5) Everything should run smoothly! :)

