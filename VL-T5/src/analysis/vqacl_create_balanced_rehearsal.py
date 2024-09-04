import os
import numpy as np
import json
import pandas as pd
from tqdm import *
import torch
import random
from collections import defaultdict, Counter
import sys
sys.path.insert(0, '../')
from Question_type import *
from tqdm import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from src.analysis.create_balanced_rehearsal import *

def _load_data(root, task_idx):
	each_memory = 20000
	json_path = root + All_task[task_idx] + '.json'
	print(f"Loading data for {All_task[task_idx]}")
	Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
	with open(json_path, 'r') as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	
	each_memory_for_cate = int(each_memory / len(Category_splits))
	for cate in Category_splits:
		num = 0
		for _d in data_info_dicts:
			img_id = _d['img_id']
			if img_id in ImgId_cate_map:
				if ImgId_cate_map[img_id] in Category_splits[cate]:
					Examplar_set[cate].append(_d)
					num += 1
					if num >= each_memory_for_cate:
						break
	All_data = []
	for key in Examplar_set:
		All_data += Examplar_set[key]
	
	return All_data

if __name__ == "__main__":
	strategy = 'none'
	mem_sizes = [5000]
	n_clusters = 7
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9)) 
	for mem_size in mem_sizes:
		print(f"Memory Size is {mem_size}") 
		task = All_task[task_idx]
		method = 'no_ents'
		split = int(mem_size / task_idx)
		root = f"../datasets/vqa/Partition_Q_V2_no_ents_past/karpathy_train_"
		data = _load_data(root, task_idx)
		print(f"Number of data points present in original data {len(data)}")
		if strategy == 'classifer':
			rehearsal_data = balanced_data_via_classifier(data, task, split, All_task, name='vqacl')
			balance_status = 'balanced'
		elif strategy == 'cluster':
			rehearsal_data = balanced_data_via_clustering(data, task, split, All_task, name='vqacl', n_clusters=n_clusters)
			balance_status = f'cluster_balanced_{n_clusters}'
		else:
			rehearsal_data = unbalanced_data(data, task, split, All_task)
			balance_status = 'unbalanced'
		print(f"No. of samples present in {task} data file:{len(rehearsal_data)}")
		savepath = root + All_task[task_idx] + '.json'
		savepath_suffix = f"{float(mem_size/1000)}k" if mem_size != 10000 else "10k"
		savepath = savepath.replace('.json', f'_{balance_status}_{savepath_suffix}.json')
		print(f"saving file @ {savepath}")
		with open(savepath, 'w') as f:
			json.dump(rehearsal_data, f, indent=4)

	print("####### Finished #######")
