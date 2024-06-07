import os
import numpy as np
import json
import pandas as pd
from tqdm import *
import torch
import random
from collections import defaultdict, Counter
import sys
sys.path.insert(0, '../')
from Question_type import *
from tqdm import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier, cat_dict
from src.analysis.question_distribution import  get_question_dist
from src.analysis.create_biased_rehearsal import classify_questions, sample_by_predicted_labels



def _load_ckpt(classifier, sub_task):
	ckpt_path =f'ckpt_vqacl/{sub_task}.pth'
	if os.path.exists(ckpt_path):
		print(f"Loading existsing checkpoint @ {ckpt_path}")
		ckpt = torch.load(f'{ckpt_path}', map_location=device)
		classifier.load_state_dict(ckpt)
	else:
		print("No ckpt found")
	return classifier

import torch
from tqdm import tqdm
import random

def unbalanced_data(data, task, split):
	# Define the root path for the captions
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# Find the task index and prepare to handle the data up to that task
	task_idx = All_task.index(task)
	final_data = []
	input_dim = 768
	hidden_dim = 256
	# Iterate through each task up to the current task index
	for i in range(task_idx):
		sub_task = All_task[i]
		output_dim = len(cat_dict_vqacl[sub_task])
		classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
		classifier = _load_ckpt(classifier, sub_task)
		question_key = f'Q_{sub_task}'
		answer_key = f'A_{sub_task}'

		# Filter out data items that don't have both the question_key and answer_key
		sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
		# Iterate over each filtered data item to process questions and answers
		new_data, sub_task_questions = [], []
		for datum in tqdm(sub_data):
			new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
			questions = datum.get(question_key, [])
			answers = datum.get(answer_key, [])
			# Create QA pairs excluding 'room' from answers
			qa_pairs = [(q, a) for q, a in zip(questions, answers) if a != 'room']

			# Select a random QA pair if available and add to new_data
			if qa_pairs:
				question, answer = random.choice(qa_pairs)
				new_datum['Q'] = question
				new_datum['A'] = ' '.join(answer.split(' ')[:2])
				sub_task_questions.append(question)
				new_data.append(new_datum)
		random.shuffle(new_data)
		new_data = new_data[:split]
		# Append directly without sampling or balancing the data
		new_questions={}
		new_questions[sub_task] = [datum['Q'] for datum in new_data]
		label_stats = get_question_dist(classifier, new_questions, sub_task)
		print(label_stats)
		final_data.extend(new_data)

	return final_data


def balanced_data(data, task, split):
	with open(f'metrics/{task}_question_dist.json', 'r') as f:
		desired_counts = json.load(f)
	# Define the root path for the captions
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# Find the task index and prepare to handle the data up to that task
	task_idx = All_task.index(task)
	final_data = []
	input_dim = 768
	hidden_dim = 256
	# Iterate through each task up to the current task index
	for i in range(task_idx):
		sub_task = All_task[i]
		desired_task_counts = desired_counts[sub_task]['balanced']
		desired_task_counts = {int(k):int(v*split/100.) for k,v in desired_task_counts.items()}
		output_dim = len(cat_dict_vqacl[sub_task])
		classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
		classifier = _load_ckpt(classifier, sub_task)
		question_key = f'Q_{sub_task}'
		answer_key = f'A_{sub_task}'

		# Filter out data items that don't have both the question_key and answer_key
		sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
		count = 0
		# Iterate over each filtered data item to process questions and answers
		new_data, sub_task_questions = [], []
		for datum in tqdm(sub_data):
			new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
			questions = datum.get(question_key, [])
			answers = datum.get(answer_key, [])
			# Create QA pairs excluding 'room' from answers
			qa_pairs = [(q, a) for q, a in zip(questions, answers)]

			# Select a random QA pair if available and add to new_data
			if qa_pairs:
				question, answer = random.choice(qa_pairs)
				new_datum['Q'] = question
				new_datum['A'] = ' '.join(answer.split(' ')[:2])
				sub_task_questions.append(question)
				new_data.append(new_datum)
		predictions = classify_questions(classifier, sub_task_questions, sub_task)
		new_data = sample_by_predicted_labels(new_data, predictions, desired_task_counts, total_target=split)
		new_questions={}
		new_questions[sub_task] = [datum['Q'] for datum in new_data]
		label_stats = get_question_dist(classifier, new_questions, sub_task)
		print(label_stats)
		final_data.extend(new_data)
	
	return final_data

def _load_data(root, task_idx):
	each_memory = 20000
	json_path = root + All_task[task_idx] + '.json'
	print(f"Loading data for {All_task[task_idx]}")
	Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
	with open(json_path, 'r') as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	
	each_memory_for_cate = int(each_memory / len(Category_splits))
	for cate in Category_splits:
		num = 0
		for _d in data_info_dicts:
			img_id = _d['img_id']
			if img_id in ImgId_cate_map:
				if ImgId_cate_map[img_id] in Category_splits[cate]:
					Examplar_set[cate].append(_d)
					num += 1
					if num >= each_memory_for_cate:
						break
	All_data = []
	for key in Examplar_set:
		All_data += Examplar_set[key]
	
	return All_data

if __name__ == "__main__":
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9)) 
	for task_idx in range(1,2):
		task =All_task[task_idx]
		split = int(5000/task_idx)
		root = "../datasets/vqa/Partition_Q_V2_no_ents/karpathy_train_"
		data = _load_data(root, task_idx)
		print(f"Number of data points present in original data {len(data)}")
		rehearsal_data = unbalanced_data(data, task, split)
		print(f"No. of samples present in {task} data file:{len(rehearsal_data)}")
		savepath = root + All_task[task_idx] + '.json'
		savepath = savepath.replace('.json', 'unbalanced.json')
		with open(savepath, 'w') as f:
			json.dump(rehearsal_data, f, indent=4)
		print("#######Finished######")
