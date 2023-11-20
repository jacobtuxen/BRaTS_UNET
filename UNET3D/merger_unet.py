import os

# List of source directories
source_dirs = ["UNET3D/unet_model/", "UNET3D/utils/", "UNET3D/"]
output_file_path = "UNET3D/HPC_FILE/output_merged_file.py"

# Specify the files to be read first
first_files = [
    "UNET3D/unet_model/unet_parts.py",
    "UNET3D/unet_model/unet_model.py",
    "UNET3D/utils/dice_score.py",
    "UNET3D/utils/focal_loss.py",
    "UNET3D/utils/generalized_dice.py",
    "UNET3D/visualize.py",
]

# Specify files to be excluded
excluded_files = [
    "UNET3D/merger_unet.py",
    "UNET3D/unet_model/count_parameters.py",
    "UNET3D/unet_model/test_model.py",
]

# Get all .py files in the directories
py_files = []
for source_dir in source_dirs:
    py_files.extend([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.py')])

# Sort the files
py_files.sort()

# Ensure the specified first files are indeed in the list and then prioritize them
for first_file in reversed(first_files):  # reverse to maintain order when inserting at the front
    if first_file in py_files:
        py_files.remove(first_file)
        py_files.insert(0, first_file)

# Specify imports to ignore
ignored_imports = [
    "from unet_model.unet_model import UNet3D",
    "from utils.dice_score import multiclass_dice_coeff, dice_coeff",
    "from .unet_parts import *",
    "from data_loader import", #<-- REWRITE THIS TO MATCH DATA LOADER
    "from unet_model.unet_model import UNet",
    "from evaluate import evaluate",
    "from utils.dice_score import dice_loss",
    "from predictions import plot_predictions",
    "from UNET3D.data_loader import BrainDataset",
    "from UNET3D.visualize import visualize_model_output",
    "from utils.focal_loss import focal_loss",
    "from utils.generalized_dice import GeneralizedDiceLoss"
]

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for file_path in py_files:
        # Don't copy the output file or excluded files if they're in the list
        if file_path == output_file_path or file_path in excluded_files:
            continue
        
        # Open each file and write its contents to the output file
        with open(file_path, 'r', encoding='utf-8') as input_file:
            output_file.write(f"# Content from: {file_path}\n")
            
            # Read file contents line by line to check for ignored import statements
            for line in input_file:
                if not any(ignored_import in line for ignored_import in ignored_imports):
                    output_file.write(line)
            
            output_file.write("\n\n")
print("MERGE SUCCESFUL!")