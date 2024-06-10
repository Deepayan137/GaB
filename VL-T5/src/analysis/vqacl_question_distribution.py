import os
import torch
import numpy as np
import json
import random
from collections import Counter
from tqdm import *
import sys
sys.path.insert(0, '../')
from Question_type import *
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier, cat_dict
from src.analysis.question_distribution import classify_questions, cluster_questions, get_question_dist


def load_gen_data(task):
	cap_root = "../datasets/vqa/Partition_Q_V2_no_ents/"
	json_path = os.path.join(cap_root, f"karpathy_train_{task}.json")
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
			question_key = f'Q'
			if question_key in datum:
				data_[sub_task].append(datum[question_key])
	return data_

def load_orig_data(task, split, size=5000):
	json_path = os.path.join('../datasets/vqa/Partition_Q_V2',f"karpathy_{split}_{task}.json")
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



# def get_question_dist(model, questions, task, batch_size=32):
# 	model.eval()  # Ensure the model is in evaluation mode
# 	label_counts = Counter()
# 	# Loop through batches of questions
# 	for i in trange(0, len(questions[task]), batch_size):
# 		batch_questions = questions[task][i:i + batch_size]
# 		embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
# 		embeddings = torch.stack(embeddings)  # Convert list of tensors to a single tensor

# 		# Ensure the embeddings are on the same device as the model
# 		embeddings = embeddings.to(next(model.parameters()).device)
		
# 		with torch.no_grad():  # Disable gradient computation
# 			outputs = model(embeddings)  # Get the logits from the model
# 			probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
# 			predicted_labels = torch.argmax(probabilities, dim=1)  # Get the index of the max probability
# 		# Update counts for this batch
# 		label_counts.update(predicted_labels.cpu().numpy())

# 	return label_stats(label_counts)

def label_stats(label_counts):
	total = sum(label_counts.values())
	percentages = {label: (count / total * 100) for label, count in label_counts.items()}
	return percentages

if __name__ == "__main__":
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
	task = All_task[task_idx]
	created = load_gen_data(task)
	input_dim = 768
	hidden_dim = 256
	strategy = 'cluster'
	print(f"strategy is {strategy}")
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	summary_dict = {}
	for i in range(task_idx):
		sub_task = All_task[i]
		summary_dict[sub_task] = {}
		output_dim = len(cat_dict_vqacl[sub_task])
		train_data = load_orig_data(sub_task, 'train', size=20000)
		test_data = load_orig_data(sub_task, 'test', size=5000)
		if strategy == 'classify':
			classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
			classifier = _load_classifier_ckpt(classifier, sub_task, name='vqacl')
			predictions_created = classify_questions(classifier, created, sub_task)
			predictions_train = classify_questions(classifier, train_data, sub_task)
		elif strategy == 'cluster':
			if not os.path.exists(f'ckpt_vqacl/kmeans_{task}.pkl'):
				predictions_train = cluster_questions(train_data, sub_task, train=True, name='vqacl')
			predictions_test = cluster_questions(test_data, sub_task, train=False, name='vqacl')
			# predictions_created = cluster_questions(created, sub_task, train=False, name='vqacl')

		# label_counts_created = get_question_dist(predictions_created)
		label_counts_train = get_question_dist(predictions_test)
		# Store results in a more readable format
		summary_dict[sub_task] = {
			'balanced': {str(k): v for k, v in label_counts_train.items()},
			# 'unbalanced': {str(k): v for k, v in label_counts_created.items()}
		}
		
		# print(f'For task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
		print(f'For task {sub_task} the distribution of labels in the test data is {label_counts_train}')
	
	file_name = "question_dist_via_clustering.json" if strategy == 'cluster' else "question_dist.json"
	with open(os.path.join("metrics", f"{task}_{file_name}"), 'w') as f:
		json.dump(summary_dict, f, indent=4)

