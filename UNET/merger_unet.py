import os

# List of source directories
source_dirs = ["UNET/unet_model/", "UNET/utils/", "UNET/"]
output_file_path = "UNET/HPC_FILE/output_merged_file.py"

# Specify the files to be read first
first_files = [
    "UNET/unet_model/unet_parts.py",
    "UNET/unet_model/unet_model.py",
    "UNET/utils/dice_score.py",
]

# Specify files to be excluded
excluded_files = [
    "UNET/merger_unet.py",
    # "UNET/test.py",
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
    "from unet_model.unet_model import UNet",
    "from utils.dice_score import multiclass_dice_coeff, dice_coeff",
    "from .unet_parts import *",
    "from data_loader import", #<-- REWRITE THIS TO MATCH DATA LOADER
    "from unet_model.unet_model import UNet",
    "from evaluate import evaluate",
    "from utils.dice_score import dice_loss"
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