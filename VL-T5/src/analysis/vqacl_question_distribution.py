import os
import torch
import numpy as np
import json
from collections import Counter
from tqdm import *
import sys
sys.path.insert(0, '../')
from Question_type import *
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier, cat_dict


def _load_orig_exemplar(task):
	savepath = f'snap/naiveblip_cl_no_ents/{task}_LAST.pth'
	ckpt = torch.load(savepath, map_location='cpu')
	task_idx = All_task.index(task)
	split=5000//task_idx
	import pdb;pdb.set_trace()
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
			if 'sent' in datum:
				sent = datum['sent']
			elif 'question' in datum:
				sent = datum['question']
			new_data[key].append(sent)
	return new_data


def load_gen_data(task):
	cap_root = "../datasets/vqa/Partition_Q_V2_subset_ST/"
	json_path = os.path.join(cap_root, f"karpathy_train_{task}.json")
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
	json_path = os.path.join('../datasets/vqa/Partition_Q_V2',f"karpathy_{split}_{task}.json")
	with open(json_path, 'r') as f:
		data = json.load(f)
	data_ = {}
	data_[task] = []
	for datum in data:
		if 'sent' in datum:
			sent = datum['sent']
		elif 'question' in datum:
			sent = datum['question']
		data_[task].append(sent)
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
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9))
	task = All_task[task_idx]
	created = load_gen_data(task)
	input_dim = 768
	hidden_dim = 256
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	summary_dict = {}
	for i in range(task_idx):
		sub_task = All_task[i]
		summary_dict[sub_task] = {}
		output_dim = len(cat_dict_vqacl[sub_task])
		classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
		test_data = load_orig_data(sub_task, 'test')
		ckpt_path = f'ckpt_vqacl/{sub_task}.pth'
		if os.path.exists(f'{ckpt_path}'):
			print(f"Loading existsing checkpoint from {ckpt_path}")
			ckpt = torch.load(f'ckpt_vqacl/{sub_task}.pth', map_location=device)
			classifier.load_state_dict(ckpt)
		# label_counts_examplar = get_question_dist(classifier, examplar, sub_task)
		label_counts_created = get_question_dist(classifier, created, sub_task)
		label_counts_test = get_question_dist(classifier, test_data, sub_task)
		# print(f'for task {sub_task} the distribution of labels in real rehearsal data is {label_counts_examplar}')
		label_counts_test = {str(k):v for k,v in label_counts_test.items()}
		label_counts_created = {str(k):v for k,v in label_counts_created.items()}
		summary_dict[sub_task]['balanced'] = label_counts_test
		summary_dict[sub_task]['unbalanced'] = label_counts_created
		print(f'for task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
		print(f'for task {sub_task} the distribution of labels in the test data is {label_counts_test}')
	with open(f"metrics/{task}_question_dist.json", 'w') as f:
		json.dump(summary_dict, f, indent=4)

