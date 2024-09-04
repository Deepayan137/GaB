import os
import torch
import numpy as np
import json
import pickle
from collections import Counter, defaultdict
from tqdm import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier
from src.analysis.question_distribution import *
sys.path.insert(0, '../')
from Question_type import *

def _load_classifier_ckpt(classifier, sub_task, name='sgvqa'):
	if name == 'sgvqa':
		ckpt_path =f'ckpt/{sub_task}.pth'
	else:
		ckpt_path = f'ckpt_vqacl/{sub_task}.pth'
	if os.path.exists(ckpt_path):
		print(f"Loading existsing checkpoint @ {ckpt_path}")
		ckpt = torch.load(f'{ckpt_path}', map_location=device)
		classifier.load_state_dict(ckpt)
	else:
		print("No ckpt found")
	return classifier

def load_orig_data(task, split, size=5000):
	if split == 'train':
		task_idx = All_task.index(task)
		each_memory = int(size / task_idx)
		orig_data = {}
		for t in range(task_idx):
			tsk = All_task[t]
			orig_data[tsk] = []
			json_path = os.path.join('../datasets/vqa/Partition_Q_V2/', f"karpathy_train_{tsk}.json")
			with open(json_path, 'r') as f:
				data = json.load(f)
			random.shuffle(data)
			orig_data[tsk] = data[:each_memory]
		data_ = {}
		for task, all_data in orig_data.items():
			data_[task] = []
			for datum in all_data:
				data_[task].append(datum['sent'])
	else:
		json_path = os.path.join('../datasets/vqa/Partition_Q_V2/', f"karpathy_train_{task}.json")
		with open(json_path, 'r') as f:
			data = json.load(f)
		data_ = {f'{task}':[]}
		for datum in data:
			data_[task].append(datum['sent'])
	return data_

def load_gen_data(task, json_path):
	# print(f"Loading original file from @ {json_path}")
	with open(json_path, 'r') as f:
		data = json.load(f)
	task_idx = All_task.index(task)
	data_ = {}
	for i in range(task_idx):
		sub_task = All_task[i]
		if sub_task not in data_:
			data_[sub_task] = []
		for datum in data:
			question_key = f'Q_{sub_task}'
			if question_key in datum:
				data_[sub_task].append(datum[question_key])
	return data_


suffix_mapping = {
			1000: '_1.0k',
			5000: '_5.0k',
			2500: '_2.5k',
			10000: '_10k',
		}

def build_data_info_path(root, task, mem_size, strategy, n_clusters=None):
		# Define the suffix based on M

		# Determine the balance type
		if strategy == "classifier":
			balance_type = "balanced"
		elif strategy == "cluster":
			balance_type = f"cluster_balanced_{n_clusters}"
		else:
			balance_type = "unbalanced"

		# Get the appropriate suffix for the given M, default to an empty string if not found
		suffix = suffix_mapping.get(mem_size, '')

		# Construct the file path
		file_name = f"karpathy_train_{task}_{balance_type}{suffix}.json"
		data_info_path = os.path.join(root, file_name)

		return data_info_path

if __name__ == "__main__":
	mem_sizes = [5000]
	strategy = 'classifier'
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 4))
	n_clusters = 7
	task = All_task[task_idx]  # Simplified to only run for the first task
	root = "../datasets/vqa/Partition_Q_V2_no_ents/"
	for mem_size in mem_sizes:
		balanced_json_path = build_data_info_path(root, task, mem_size, strategy, n_clusters=n_clusters)
		print(f"Loading data from {balanced_json_path}")
		balanced = load_gen_data(task, balanced_json_path)
		unbalanced_json_path = build_data_info_path(root, task, mem_size, 'none', n_clusters=None)
		print(f"Loading data from {unbalanced_json_path}")
		unbalanced = load_gen_data(task, unbalanced_json_path)
		orig_data =  load_orig_data(task, 'train', size=mem_size)
		input_dim = 768
		hidden_dim = 256
		summary_dict = defaultdict(dict)
		# Loop over all tasks, but limit to the first for simplification
		for i, sub_task in enumerate(All_task[:task_idx]):
			output_dim = len(cat_dict_vqacl[sub_task])
			# test_data = load_orig_data(sub_task, 'val', size=mem_size)
			# Initialize classifier if the strategy is to classify
			if strategy == 'classifier':
				classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
				classifier = _load_classifier_ckpt(classifier, sub_task, name='vqacl')
				preds_balanced = classify_questions(classifier, balanced, sub_task)
				preds_unbalanced = classify_questions(classifier, unbalanced, sub_task)
				preds_orig = classify_questions(classifier, orig_data, sub_task)
				# preds_test = classify_questions(classifier, test_data, sub_task)
			elif strategy == 'cluster':
				filename = f'ckpt_vqacl/kmeans_{sub_task}_{n_clusters}.pkl'
				if not os.path.exists(filename):
					print(f'{filename} does not exist. Clustering...')
					preds_orig = cluster_questions(orig_data, sub_task, train=True, filename=filename)
				# preds_test = cluster_questions(test_data, sub_task, train=False, filename=filename)
				preds_balanced = cluster_questions(balanced, sub_task, filename=filename)
				preds_unbalanced = cluster_questions(unbalanced, sub_task, filename=filename)
				preds_orig = cluster_questions(orig_data, sub_task, filename=filename)
			label_counts_orig = get_question_dist(preds_orig)
			label_counts_unbalanced = get_question_dist(preds_unbalanced)
			label_counts_balanced = get_question_dist(preds_balanced)
			# label_counts_test = get_question_dist(preds_test)
			# Store results in a more readable format
			summary_dict[sub_task] = {
				'real': {str(k): v for k, v in label_counts_orig.items()},
				'balanced': {str(k): v for k, v in label_counts_balanced.items()},
				'unbalanced': {str(k): v for k, v in label_counts_unbalanced.items()},
				# 'test': {str(k): v for k, v in label_counts_test.items()}
			}

			# print(f'For task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
			# print(f'For task {sub_task} the distribution of labels in the test data is {label_counts_train}')

		# Save results based on strategy
		suffix = suffix_mapping.get(mem_size, '')
		file_name = f"question_dist_via_clustering_{n_clusters}.json" if strategy == 'cluster' else "question_dist_qtype.json"
		with open(os.path.join("vqacl_qdists", f"vqacl_{task}{suffix}_{file_name}"), 'w') as f:
			json.dump(summary_dict, f, indent=4)

