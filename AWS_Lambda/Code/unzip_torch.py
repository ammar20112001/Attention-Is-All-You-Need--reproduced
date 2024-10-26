import os
import sys
import zipfile


torch_dir_to = "/tmp/torch"
torch_dir_from = "torch.zip"

if not os.path.exists(torch_dir_to):
    os.mkdir(torch_dir_to)
    # Unzip Torch dependency to /tmp/ directory
    zipfile.ZipFile(torch_dir_from, "r").extractall(torch_dir_to)

# Add /tmp/torch to environment
sys.path.append(torch_dir_to)