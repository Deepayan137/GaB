import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# from src.param import parse_args
from random import sample
import numpy as np
import random
random.seed(21)
import json
import sys
sys.path.insert(0, '../')
from Question_type import *


if __name__ == "__main__":
	# args = parse_args()
	root = '../datasets/vqa/Partition_Generated'
	
	for task in All_task[1:]:
		with open(f'{root}/{task}_gen_qa_pairs.json') as f:
			data = json.load(f)
		mydict = {}
		print(f"Length of data before:{len(data)}")
		for k, v in data.items():
			mydict[k]=[]
			if k.startswith('COCO'):
				for item in v:
					if 'Q' in item and 'A' in item:
						mydict[k].append(item)
					else:
						print(item)
		print(f"Length of data after removal:{len(mydict)}")
		with open(f'{root}/{task}_gen_qa_pairs.json', 'w') as f:
			json.dump(mydict, f)