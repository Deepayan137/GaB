import os
import json
import random
import sys
sys.path.insert(0, '../')
from Question_type import *

def qtypes():
	dest_root = '../datasets/vqa/Partition_Q_V2'
	qtype_dict = {}
	qtype_ex = {}
	for task_idx, task in enumerate(All_task):
		qtype_dict[task] = {}
		qtype_ex[task] = {}
		data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{All_task[task_idx]}.json')
		count=0
		with open(data_info_path, 'r') as f:
			data_info_dicts = json.load(f)
		import pdb;pdb.set_trace()
		for _d in data_info_dicts:
			qtype = _d['question_type']
			if qtype not in qtype_dict[task]:
				qtype_dict[task][qtype] = 0
				qtype_ex[task][qtype] = [_d['sent']]
			else:
				qtype_dict[task][qtype] += 1
				count+=1
				if len(qtype_ex[task][qtype]) <= 5:
					qtype_ex [task][qtype].append(_d['sent'])
		qtype_dict[task]['all'] = count
		qtype_dict[task] = dict(sorted(qtype_dict[task].items(), key=lambda item: item[1], reverse=True))
	with open("../datasets/vqa/Qtypes.json", 'w') as f:
		json.dump(qtype_dict, f, indent=4)		
	with open("../datasets/vqa/Qexamples.json", 'w') as f:
		json.dump(qtype_ex, f, indent=4)


if __name__ == "__main__":
	# mem_status()
	qtypes()