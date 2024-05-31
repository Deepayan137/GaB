import os
import torch
import numpy as np
import json
from collections import Counter
from tqdm import *
All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from src.analysis.qtype_sim import get_embedding, QuestionTypeClassifier


def _load_orig_exemplar(task):
	savepath = f'snap/naiveblip_sgvqa_mem_new/{task}_BEST.pth'
	ckpt = torch.load(savepath, map_location='cpu')
	task_idx = All_task.index(task)
	split=5000//task_idx
	# Initialize a dictionary to hold the Counters for each sub_task
	data = {}
	for i in range(task_idx):
		sub_task = All_task[i]
		# Make sure the sub_task key exists in the ckpt['exemplar'] to avoid KeyError
		if sub_task in ckpt['examplar']:
			data[sub_task] = ckpt['examplar'][sub_task][:split]
	new_data={}
	for key in data:
		if key not in new_data:
			new_data[key]=[]
		for datum in data[key]:
			new_data[key].append(datum['question'])
	return new_data


def load_gen_data(task):
	cap_root = "../datasets/npy_self/function/"
	json_path = os.path.join(cap_root, f"fcl_mmf_{task}_train.json")
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

def load_orig_data(task, split):
	np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_{split}.npy")
	data = np.load(np_path, allow_pickle=True)
	data_ = {}
	data_[task] = []
	for datum in data:
		data_[task].append(datum['question'])
	return data_



def get_question_dist(model, questions, task, batch_size=32):
	model.eval()  # Ensure the model is in evaluation mode
	label_counts = Counter()

	# Loop through batches of questions
	for i in trange(0, len(questions[task]), batch_size):
		batch_questions = questions[task][i:i + batch_size]
		embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
		embeddings = torch.stack(embeddings)  # Convert list of tensors to a single tensor

		# Ensure the embeddings are on the same device as the model
		embeddings = embeddings.to(next(model.parameters()).device)
		
		with torch.no_grad():  # Disable gradient computation
			outputs = model(embeddings)  # Get the logits from the model
			probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
			predicted_labels = torch.argmax(probabilities, dim=1)  # Get the index of the max probability
		# Update counts for this batch
		label_counts.update(predicted_labels.cpu().numpy())

	return label_stats(label_counts)

def label_stats(label_counts):
	total = sum(label_counts.values())
	percentages = {label: (count / total * 100) for label, count in label_counts.items()}
	return percentages

if __name__ == "__main__":
	task = 'logical'
	task_idx = All_task.index(task)
	examplar = _load_orig_exemplar(task)
	created = load_gen_data(task)
	input_dim = 768
	hidden_dim = 256
	output_dim = 20 if task != 'logical' else 23
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
	for i in range(task_idx):
		sub_task = All_task[i]
		test_data = load_orig_data(sub_task, 'val')
		if os.path.exists(f'ckpt/{sub_task}.pth'):
			print("Loading existsing checkpoint")
			ckpt = torch.load(f'ckpt/{sub_task}.pth', map_location=device)
			classifier.load_state_dict(ckpt)
		label_counts_examplar = get_question_dist(classifier, examplar, sub_task)
		label_counts_created = get_question_dist(classifier, created, sub_task)
		label_counts_test = get_question_dist(classifier, test_data, sub_task)
		print(f'for task {sub_task} the distribution of labels in real rehearsal data is {label_counts_examplar}')
		print(f'for task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
		print(f'for task {sub_task} the distribution of labels in the test data is {label_counts_test}')
		# import pdb;pdb.set_trace()


