import os
import torch
import multiprocessing
from tqdm import tqdm

def is_file_broken(filepath):
    try:
        torch.load(filepath, map_location=torch.device('cpu'))
        return None  # File is not broken
    except Exception as e:
        return filepath  # Return the path of the broken file

def check_pt_files(files):
    broken_files = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(is_file_broken, files), total=len(files)):
            if result is not None:
                broken_files.append(result)
    return broken_files

def get_all_pt_files(directory):
    pt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    return pt_files

def main():
    directory = "/home/ycevan/s3prl/features/Salmonn13B"  # Replace with your directory path
    pt_files = get_all_pt_files(directory)
    
    if pt_files:
        broken_files = check_pt_files(pt_files)
        if broken_files:
            print("Broken files:")
            for file in broken_files:
                print(file)
        else:
            print("No broken files found.")
    else:
        print("No .pt files found in the directory.")

if __name__ == "__main__":
    main()
