import os
import torch
import numpy as np
import json
import argparse
import random
from collections import Counter
from tqdm import *
import sys
sys.path.insert(0, '../')
from Question_type import *
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier, cat_dict
from src.analysis.question_distribution import classify_questions, cluster_questions, get_question_dist, _load_classifier_ckpt


def load_gen_data(task, root):
	# cap_root = "../datasets/vqa/Partition_Q_V2_no_ents_past/"
	json_path = os.path.join(root, f"karpathy_train_{task}.json")
	print(f"Loading original file from @ {json_path}")
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

def load_orig_data(task, root, split, size=5000):
	json_path = os.path.join(root, f"karpathy_{split}_{task}.json")
	with open(json_path, 'r') as f:
		data = json.load(f)
	data_ = {}
	data_[task] = []
	if split == 'train':
		random.shuffle(data)
		data = data[:size]
	for datum in data:
		if 'sent' in datum:
			sent = datum['sent']
		elif 'question' in datum:
			sent = datum['question']
		data_[task].append(sent)
	return data_


def label_stats(label_counts):
	total = sum(label_counts.values())
	percentages = {label: (count / total * 100) for label, count in label_counts.items()}
	return percentages

def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--root_path", default="../datasets/vqa/Partition_Q_V2/", type=str)
    parser.add_argument("--gen_path", default="../datasets/vqa/Partition_Q_V2_test/", type=str)
    parser.add_argument("--coco_path", default="/mhug/mhug-dataset/COCO", type=str)
    parser.add_argument("--savepath", default="../ckpt_vqacl", type=str)
    parser.add_argument("--strategy", type=str, default='cluster')
    parser.add_argument("--n_clusters", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	input_dim = 768
	hidden_dim = 256
	n_clusters = args.n_clusters
	strategy = args.strategy
	print(f"strategy is {strategy}")
	os.makedirs(args.savepath, exist_ok=True)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9))
	for task_idx in range(1, len(All_task)):
		task = All_task[task_idx]
		created = load_gen_data(task, args.gen_path)
		summary_dict = {}
		for i in range(task_idx):
			sub_task = All_task[i]
			summary_dict[sub_task] = {}
			output_dim = len(cat_dict_vqacl[sub_task])
			train_data = load_orig_data(sub_task, args.root_path, 'train', size=20000)
			test_data = load_orig_data(sub_task, args.root_path, 'train', size=5000)
			if strategy == 'classifier':
				classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
				classifier = _load_classifier_ckpt(classifier, sub_task, name='vqacl')
				predictions_created = classify_questions(classifier, created, sub_task)
				predictions_train = classify_questions(classifier, test_data, sub_task)
			elif strategy == 'cluster':
				filename = os.path.join(args.savepath, f'kmeans_{sub_task}_{n_clusters}.pkl')
				if not os.path.exists(filename):
					print(f'No {filename} found')
					predictions_train = cluster_questions(train_data, sub_task, train=True, filename=filename, n_clusters=n_clusters)
				predictions_created = cluster_questions(created, sub_task, train=False, filename=filename, n_clusters=n_clusters)
				predictions_train = cluster_questions(test_data, sub_task, train=False, filename=filename, n_clusters=n_clusters)
				
			label_counts_created = get_question_dist(predictions_created)
			label_counts_train = get_question_dist(predictions_train)
			summary_dict[sub_task] = {
				'balanced': {str(k): v for k, v in label_counts_train.items()},
				'unbalanced': {str(k): v for k, v in label_counts_created.items()}
			}
			
			print(f'For task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
			print(f'For task {sub_task} the distribution of labels in the test data is {label_counts_train}')
		
		file_name = f"question_dist_via_clustering_{n_clusters}.json" if strategy == 'cluster' else "question_dist.json"
		dest_dir = '../metrics'
		os.makedirs(dest_dir, exist_ok=True)
		with open(os.path.join(dest_dir, f"{task}_{file_name}"), 'w') as f:
			json.dump(summary_dict, f, indent=4)

