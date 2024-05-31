import os
import numpy as np
import json
import pandas as pd
from tqdm import *
import torch
import random
from collections import defaultdict, Counter

All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from tqdm import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier
from src.analysis.question_distribution import  get_question_dist
from src.analysis.create_biased_rehearsal import classify_questions, sample_by_predicted_labels


desired_counts = {
	"attribute":{
		"object":{
	    0: int(5000 * 0.4708),
	    1: int(5000 * 0.34),
	    2: int(5000 * 0.196),
	    },
	},
	"relation":{
		"object":{
		0: int(2500 * 0.47),
		1: int(2500 * 0.33),
		2: int(2500 * 0.20),
		},
		"attribute":{
		8: int(2500 * 0.3932),
		6: int(2500 * 0.0708),
		7: int(2500 * 0.0820),
		1: int(2500 * 0.2596),
		0: int(2500 * 0.1560)
		}
	}
}


def _load_ckpt(classifier, sub_task):
	if os.path.exists(f'ckpt/{sub_task}.pth'):
		print("Loading existsing checkpoint")
		ckpt = torch.load(f'ckpt/{sub_task}.pth', map_location=device)
		classifier.load_state_dict(ckpt)
	else:
		print("No ckpt found")
	return classifier

def balanced_data(task, split, classifier):
	# Define the root path for the captions
	cap_root = "../datasets/npy_cap_test/function/"
	json_path = os.path.join(cap_root, f"fcl_mmf_{task}_train.json")

	# Load JSON data
	with open(json_path, 'r') as f:
		data = json.load(f)

	# Find the task index and prepare to handle the data up to that task
	task_idx = All_task.index(task)
	final_data = []

	# Iterate through each task up to the current task index
	for i in range(task_idx):
		sub_task = All_task[i]
		classifier = _load_ckpt(classifier, sub_task)
		question_key = f'Q_ents_{sub_task}'
		answer_key = f'A_ents_{sub_task}'

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
			qa_pairs = [(q, a) for q, a in zip(questions, answers) if a != 'room']

			# Select a random QA pair if available and add to new_data
			if qa_pairs:
				question, answer = random.choice(qa_pairs)
				new_datum['Q'] = question
				new_datum['A'] = ' '.join(answer.split(' ')[:2])
				sub_task_questions.append(question)
				new_data.append(new_datum)
		predictions = classify_questions(classifier, sub_task_questions, sub_task)
		new_data = sample_by_predicted_labels(new_data, predictions, desired_counts[task][sub_task])
		new_questions={}
		new_questions[sub_task] = [datum['Q'] for datum in new_data]
		label_stats = get_question_dist(classifier, new_questions, sub_task)
		print(label_stats)
		final_data.extend(new_data)
	
	return final_data

if __name__ == "__main__":
	
	input_dim = 768
	hidden_dim = 256
	output_dim = 20
	classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)


	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 2)) 
	task =All_task[task_idx]
	split = int(5000/task_idx)
	rehearsal_data =  balanced_data(task, split, classifier)
	print(f"No. of samples present in {task} data file:{len(rehearsal_data)}")
	savepath = os.path.join(f'../datasets/npy_balanced/function', f'fcl_mmf_{task}_train.json')
	with open(savepath, 'w') as f:
		json.dump(rehearsal_data, f, indent=4)
