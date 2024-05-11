import os
import json
import numpy as np
import sys
sys.path.insert(0, '../')
from Question_type import Sg_task
from tqdm import *


if __name__ == "__main__":
	task_type = "function"
	path = f"../datasets/npy/{task_type}/"
	All_tasks = Sg_task[task_type]['oarlks']
	task_idx = 3
	for task in tqdm(All_tasks):
		np_path = os.path.join(path, f"fcl_mmf_{task}_train.npy")
		data = np.load(np_path, allow_pickle=True)
		qtype_dict = {}
		for data_dict in data:
			qtype = data_dict['raw_question_type']
			question = data_dict['question']
			answer = data_dict['answer']
			qa = f"{question}{answer}"
			if str(qtype) not in qtype_dict:
				qtype_dict[str(qtype)] = []
			qtype_dict[str(qtype)].append(qa)
			if len(qtype_dict[str(qtype)]) >= 100:
				break
		with open(f"../datasets/npy/{task_type}/{task}_qtype.json", 'w') as f:
			json.dump(qtype_dict, f, indent=4)


