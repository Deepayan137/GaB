import os
import sys
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.image import imread
import shutil
# sys.path.insert(0, '../')
# from Question_type import Sg_task, All_task

random.seed(60)
qkeys = ['Q_q_recognition', 'Q_q_location', 'Q_q_judge', 'Q_q_commonsense', 'Q_q_count', 'Q_q_action',
'Q_q_color', 'Q_q_type', 'Q_q_subcategory']
akeys = ['A_q_recognition', 'A_q_location', 'A_q_judge', 'A_q_commonsense', 'A_q_count', 'A_q_action',
'A_q_color', 'A_q_type', 'A_q_subcategory']

def read_json(json_file):
	with open(json_file, 'r') as f:
		data = json.load(f)
	return data

def common_ids(data_gab, data_qtype):
	gab_ids = [datum['img_id'] for datum in data_gab]
	qtype_ids = [datum['img_id'] for datum in data_qtype]
	gab_ids, qtype_ids = set(gab_ids), set(qtype_ids)
	return list(gab_ids.intersection(qtype_ids))

def get_qa(data, ids):
	qa_dict = {}
	for datum in data:
		img_id = datum.get('img_id')
		if img_id in ids:
			for qk in qkeys:
				if qk in datum.keys():
					qa_dict[img_id] = {'Q':datum[qk][0]}
			for ak in akeys:
				if ak in akeys:
					if ak in datum.keys():
						qa_dict[img_id]['A']=datum[ak][0]
	return qa_dict					

def get_ques(qa_gab, qa_qtype):
	diff = []
	for img_id, qa in qa_gab.items():
		q_gab, q_qtype = qa['Q'], qa_qtype[img_id]['Q']
		a_gab, a_qtype = qa['A'], qa_qtype[img_id]['A']
		dictum = {'img_id':img_id, 'q_gab':q_gab, 'q_qtype':q_qtype, 
		'a_gab':a_gab, 'a_qtype':a_qtype}
		diff.append(dictum)
	return diff


def plot_grid(qa_data, k):
	qa_data_sub = random.sample(qa_data, 9)
	dest = f'viz/supp_plots/{k}'
	os.makedirs(dest, exist_ok=True)
	# Set up the figure and axes
	fig, axes = plt.subplots(3, 3, figsize=(15, 10))
	font_size = 10  # Smaller font size
	for i, ax in enumerate(axes.flat):
		# Read the image
		img_id = qa_data_sub[i]['img_id']
		img_dir = img_id.split('_')[1]
		img_path = os.path.join('../datasets/COCO/', img_dir, img_id+'.jpg')
		q_gab = qa_data_sub[i]['q_gab']
		a_gab = qa_data_sub[i]['a_gab']
		q_qtype = qa_data_sub[i]['q_qtype']
		a_qtype = qa_data_sub[i]['a_qtype']
		
		shutil.copy(img_path, dest)
		img = imread(img_path)
		label = f"Q:{q_qtype}\nA:{a_qtype}"
		# \nGaB (Ours)->Q:{q_gab}\nA:{a_gab}"
		# Show the image
		ax.imshow(img)

		# Set title (label text)
		ax.set_title(label, fontsize=font_size)

		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
	random_number = random.randint(1000, 9999)
	plt.savefig(f"viz/supp_plots/grid_qual_{k}.png")
	# Adjust layout to prevent overlap
	plt.tight_layout(pad=0.5)
	plt.show()
	plt.close()

if __name__ == '__main__':
	root = '../datasets/vqa/'
	gab_file = os.path.join(root, 'Partition_Q_V2_no_ents/karpathy_train_q_causal.json')
	qtype_file = os.path.join(root, 'Partition_Q_V2_qtype/karpathy_train_q_causal.json')
	data_gab = read_json(gab_file)
	data_qtype = read_json(qtype_file)
	common_ids = common_ids(data_gab, data_qtype)
	qa_gab = get_qa(data_gab, common_ids)
	qa_qtype = get_qa(data_qtype, common_ids)
	qa_data = get_ques(qa_gab, qa_qtype)
	for k in range(10):
		plot_grid(qa_data, k)
