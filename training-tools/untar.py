import os
import argparse
import shutil
import tarfile
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Extract all .tar files in a directory and its subdirectories")
parser.add_argument("src_dir", metavar="directory", type=str, help="the directory containing the .tar files to extract")
args = parser.parse_args()

src_dir = args.src_dir

if not os.path.isdir(src_dir):
    print("Error: {} is not a directory".format(src_dir))
    exit(1)

# Get a list of all .tar files in src_dir and its subdirectories
tar_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".tar"):
            tar_files.append(os.path.join(root, file))

# Extract each .tar file and display a progress bar
overall_desc = "Extracting .tar files"
with tqdm(total=len(tar_files), desc=overall_desc, unit="files") as overall_progress:
    for tar_file in tar_files:
        with tarfile.open(tar_file, "r") as tar:
            members = tar.getmembers()
            tar_desc = "Extracting {}".format(tar_file)
            with tqdm(total=len(members), desc=tar_desc, unit="files") as tar_progress:
                for member in members:
                    # Check if the file or directory already exists before extracting
                    dst_path = os.path.join(os.path.dirname(tar_file), member.name)
                    if os.path.exists(dst_path):
                        if os.path.isdir(dst_path):
                            shutil.rmtree(dst_path)
                        else:
                            os.remove(dst_path)
                    tar.extract(member, os.path.dirname(tar_file))
                    tar_progress.update()
        overall_progress.update()

