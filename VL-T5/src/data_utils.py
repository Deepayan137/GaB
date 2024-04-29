import os
import json
import random
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from src.vqa_data_blip import VQAEvaluator, get_loader_memory, VQADataset
sys.path.insert(0, '../')
from Question_type import *
from torchvision.transforms.functional import to_tensor

def get_memory_data(args, task_idx, each_memory, Examplar_set):
	if args.use_gen_data:
		data_info_path = ('../datasets/vqa/Partition_Q_V2_subset/karpathy_train_' + f'{All_task[task_idx]}.json')
	else:
		data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{All_task[task_idx - 1]}.json')
	with open(data_info_path) as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	if args.use_class_hierarchy:
		each_memory_for_cate = int(each_memory / len(Category_splits))
		for cate in Category_splits:
			num = 0
			Examplar_set[cate].append([])
			for _d in data_info_dicts:
				img_id = _d['img_id']
				if img_id in ImgId_cate_map:
					if ImgId_cate_map[img_id] in Category_splits[cate]:
						Examplar_set[cate][task_idx - 1].append(_d)
						num += 1
						if num >= each_memory_for_cate:
							break
		print('Load from Partition_Q_v3......')
		for cate in Category_splits:
			for i in range(task_idx):
				Examplar_set[cate][i] = Examplar_set[cate][i][: each_memory_for_cate]
		All_examplar = []
		for E_set in Examplar_set:
			for task_set in Examplar_set[E_set]:
				All_examplar += task_set
	else:
		All_examplar = data_info_dicts[:each_memory]
	print("# The size of the cate Memory:", len(All_examplar))			
	return All_examplar, Examplar_set




if __name__ == "__main__":
	from src.param import parse_args
	from tqdm import *
	args = parse_args()
	args.backbone = 'blip2'
	args.train = 'karpathy_train'
	args.use_gen_data = False
	task_idx = 1
	M = 5000
	Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
	train_dset = VQADataset(args.train, True)
	for task_idx, task in enumerate(All_task):
		if task_idx >0:
			each_memory = M//task_idx
			All_examplar, Examplar_set = get_memory_data(args, task_idx, each_memory, Examplar_set)
			memory_loader = get_loader_memory(args, All_task, All_examplar, train_dset, workers=0)
			Category_splits_random = random_dic(Category_splits)
			for idx, cateGroup in enumerate(Category_splits_random):
				memory_loader_cate = memory_loader[cateGroup]
				for i, batch in enumerate(tqdm(memory_loader_cate)):
					pix = batch['pixel_values']




