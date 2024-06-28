import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import *
import torch
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier
from src.analysis.question_distribution import classify_questions, sample_by_predicted_labels, _load_classifier_ckpt, load_orig_data
import sys
sys.path.insert(0, '../')
from Question_type import qtype_dict, Sg_task


def load_orig_data(task, split, size=5000):
	np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_{split}.npy")
	data = np.load(np_path, allow_pickle=True)
	random.shuffle(data)
	data = data[:size]
	return data


def save_to_pickle(data, filename):
	with open(filename, 'wb') as file:  # 'wb' indicates that the file is opened for writing in binary mode
		pickle.dump(data, file)
	print(f"Data successfully saved to {filename}")


if __name__ == "__main__":

	input_dim = 768
	hidden_dim = 256
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 4)) 
	sequence = 'oarlks'
	All_task = Sg_task['function'][sequence]
	task = All_task[task_idx]
	new_data = {}
	count = 0
	split=5000/task_idx
	with open(f'metrics/sgvqa_{task}_question_dist.json', 'r') as f:
		desired_counts = json.load(f)

	for i in range(task_idx):
		sub_task = All_task[i]
		desired_counts_task = {int(k):int(v*split/100.) for k,v in desired_counts[sub_task]['unbalanced'].items()}
		train_data = load_orig_data(sub_task, 'train', size=20000)
		new_data[sub_task] = []
		output_dim = len(qtype_dict[sub_task])
		classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
		classifier = _load_classifier_ckpt(classifier, sub_task, name='sgvqa')
		questions = {}
		questions[sub_task] = []
		for datum in train_data:
			questions[sub_task].append(datum['question'])
		predictions = classify_questions(classifier, questions, sub_task)
		sampled_list = sample_by_predicted_labels(train_data, predictions, desired_counts_task, total_target=split)
		new_data[sub_task].extend(sampled_list)
		count += len(sampled_list)
	print(f"No. of samples present in {task} data file: {count}")
	save_to_pickle(new_data, f'../datasets/npy_biased/{task}.pkl')




