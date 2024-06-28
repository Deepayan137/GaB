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
			tsk = Sg_task['function']['oarlks'][t]
			orig_data[tsk] = []
			np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{tsk}_train.npy")
			data = np.load(np_path, allow_pickle=True)
			random.shuffle(data)
			orig_data[tsk] = data[:each_memory]
		data_ = {}
		for task, all_data in orig_data.items():
			data_[task] = []
			for datum in all_data:
				data_[task].append(datum['question'])
	else:
		np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_val.npy")
		data = np.load(np_path, allow_pickle=True)
		data_ = {f'{task}':[]}
		for datum in data:
			data_[task].append(datum['question'])
	return data_

def load_gen_data(task, json_path, size=5000):
	with open(json_path, 'r') as f:
		data = json.load(f)
	All_task = Sg_task['function']['oarlks']
	task_idx = All_task.index(task)
	data_ = {}
	mem_split = mem_size//task_idx
	for i in range(task_idx):
		sub_task = All_task[i]
		if sub_task not in data_:
			data_[sub_task] = []
		# start_idx = i * mem_split
		# end_idx = start_idx + mem_split
		# sub_data = data	[start_idx:end_idx]
		for datum in data:
			question_key = f'Q_{sub_task}'
			if question_key in datum:
				data_[sub_task].append(datum[question_key])
	return data_

suffix_mapping = {
	    5000: '_5.0k',
	    1000: '_1.0k',
	    2500: '_2.5k',
	    10000: '_10k',
	    20000: '_20k'
	}

def build_data_info_path(scenario_dir, tsk, mem_size, balance_strategy):
		# Define the suffix based on M
	

		# Determine the balance type
		if balance_strategy == "classifier":
			balance_type = "balanced"
		elif balance_strategy == "cluster":
			balance_type = "cluster_balanced"
		else:
			balance_type = "unbalanced"

		# Get the appropriate suffix for the given M, default to an empty string if not found
		suffix = suffix_mapping.get(mem_size, '')

		# Construct the file path
		file_name = f"fcl_mmf_{tsk}_train_{balance_type}{suffix}.json"
		data_info_path = os.path.join(scenario_dir, file_name)

		return data_info_path

if __name__ == "__main__":
	mem_sizes = [1000, 2500, 5000, 10000]
	strategy = 'cluster'
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 2))
	All_task = Sg_task['function']['oarlks']
	
	task = All_task[task_idx]  # Simplified to only run for the first task
	root = "../datasets/npy_no_ents/function/"
	for mem_size in mem_sizes:
		balanced_json_path = build_data_info_path(root, task, mem_size, strategy)
		print(f"Loading data from {balanced_json_path}")
		balanced = load_gen_data(task, balanced_json_path, size=mem_size)
		unbalanced_json_path = build_data_info_path(root, task, mem_size, 'none')
		print(f"Loading data from {unbalanced_json_path}")
		unbalanced = load_gen_data(task, unbalanced_json_path, size=mem_size)
		orig_data =  load_orig_data(task, 'train', size=mem_size)
		input_dim = 768
		hidden_dim = 256
		summary_dict = defaultdict(dict)
		# Loop over all tasks, but limit to the first for simplification
		for i, sub_task in enumerate(All_task[:task_idx]):
			output_dim = len(qtype_dict[sub_task])
			test_data = load_orig_data(sub_task, 'val', size=mem_size)
			# Initialize classifier if the strategy is to classify
			if strategy == 'classify':
				classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
				classifier = _load_classifier_ckpt(classifier, sub_task)
				preds_balanced = classify_questions(classifier, balanced, sub_task)
				preds_unbalanced = classify_questions(classifier, unbalanced, sub_task)
				preds_orig = classify_questions(classifier, orig_data, sub_task)
				preds_test = classify_questions(classifier, test_data, sub_task)
			elif strategy == 'cluster':
				if not os.path.exists(f'ckpt/kmeans_{sub_task}.pkl'):
					preds_orig = cluster_questions(orig_data, sub_task, train=True, name='sgvqa')
				preds_test = cluster_questions(test_data, sub_task, train=False, name='sgvqa')
				preds_balanced = cluster_questions(balanced, sub_task, name='sgvqa')
				preds_unbalanced = cluster_questions(unbalanced, sub_task, name='sgvqa')
				preds_orig = cluster_questions(orig_data, sub_task, name='sgvqa')
			label_counts_orig = get_question_dist(preds_orig)
			label_counts_unbalanced = get_question_dist(preds_unbalanced)
			label_counts_balanced = get_question_dist(preds_balanced)
			label_counts_test = get_question_dist(preds_test)
			# Store results in a more readable format
			summary_dict[sub_task] = {
				'real': {str(k): v for k, v in label_counts_orig.items()},
				'balanced': {str(k): v for k, v in label_counts_balanced.items()},
				'unbalanced': {str(k): v for k, v in label_counts_unbalanced.items()},
				'test': {str(k): v for k, v in label_counts_test.items()}
			}

			# print(f'For task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
			# print(f'For task {sub_task} the distribution of labels in the test data is {label_counts_train}')

		# Save results based on strategy
		suffix = suffix_mapping.get(mem_size, '')
		file_name = "question_dist_via_clustering.json" if strategy == 'cluster' else "question_dist.json"
		with open(os.path.join("qdists", f"sgvqa_{task}{suffix}_{file_name}"), 'w') as f:
			json.dump(summary_dict, f, indent=4)