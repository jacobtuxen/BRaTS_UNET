# BRaTS_UNET
3dUNET for BRaTS 2021 dataset

This code is written for DTU's HPC. To run this code, the user must have placed the BraTS data set in the scratch directory of the user on HPC.
The path should be as follows: ~/workX/sXXXXXX/data/... Where X is the number for the personal scratch directory. The code is currently hooked up to WandB, but to run
purely locally, change the variable USE_WANDB to False in the output_merged_file.py. To run the code, make a batch job of output_merged_file.py. Where you have changed the scratch directory
to your own. The dataloader requires a list of patients you want to train/validate. This list of patients is a .txt file that should be placed /workX/sXXXXXX/data/filenames_filtered.txt

Step by step guide to run the code:
1) Change scratch directory to your own in output_merged_file.py.
2) Change WandB api key to your own in output_merged_file.py. Or optionally set USE_WANDB to False in output_merged_file.py.
3) Update /workX/sXXXXXX/data/filenames_filtered.txt to the patients you want to train and validate.
4) Create batch job for output_merged_file.py on DTU HPC.
5) Everything should run smoothly! :)

