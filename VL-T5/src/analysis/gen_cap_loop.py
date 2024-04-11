import os
import json
import sys
import random
sys.path.insert(0, '../')
from Question_type import All_task
from src.analysis.test_context import inference_cap

if __name__ == "__main__":
	cap_dict = {}
	for task in All_task:
		path = f"../datasets/vqa/Partition_Q_V2/karpathy_train_{task}.json"
		img_dir = "../datasets/COCO/train2014/"
		f = open(path, 'r')
		data = json.load(f)
		subset = random.sample(data, 10)
		cap[task] = []
		for item in subset:
			img_name = subset['img_id']
			img_path = os.path.join(img_dir, img_name)
			cap = inference_cap(img_path)
			cap_dict[task].append((img_name, cap))
	with open('test.json', 'w')	as f:
		json.dump(cap_dict, f)