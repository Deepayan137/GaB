import os
import json

if __name__ == "__main__":
	path = '../datasets/vqa/Partition_Q_V2_subset/q_type.json '
	with open(path, 'r') as f:
		data = json.load(f)
	