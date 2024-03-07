import os
import json
from tqdm import *

def filter_blank_answers(source_path, destination_directory):
    with open(source_path, 'r') as f:
        data_dict = json.load(f)
    orig_len = len(data_dict)
    filtered_data = [datum for datum in tqdm(data_dict) if sum(datum['label'].values()) != 0.0]
    filt_len = len(filtered_data)
    print(f"Difference:{orig_len-filt_len}")
    destination_path = os.path.join(destination_directory, os.path.basename(source_path))
    with open(destination_path, 'w') as f:
        json.dump(filtered_data, f)

if __name__ == "__main__":
    source_dir = "/home/deepayan.das/projects/VQACL/datasets/vqa/Partition_Q"
    destination_dir = "/home/deepayan.das/projects/VQACL/datasets/vqa/Partition_Q_V2"

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        print(f"Filtering {filename}")
        filter_blank_answers(source_path, destination_dir)
