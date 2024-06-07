import os
import torch
import numpy as np
import json
from collections import Counter
import random
from tqdm import *
All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier, cat_dict
from sklearn.cluster import KMeans
from tqdm import trange



def load_gen_data(task):
	cap_root = "../datasets/npy_no_ents/function/"
	json_path = os.path.join(cap_root, f"fcl_mmf_{task}_train_unbalanced.json")
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

def load_orig_data(task, split):
	np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_{split}.npy")
	data = np.load(np_path, allow_pickle=True)
	random.shuffle(data)
	data = data[:5000]
	data_ = {}
	data_[task] = []
	for datum in data:
		data_[task].append(datum['question'])
	return data_


def cluster_questions(questions, task, batch_size=32, n_clusters=10):
		# Placeholder for all embeddings
		all_embeddings = []

		# Generate embeddings for all questions
		for i in trange(0, len(questions[task]), batch_size):
			batch_questions = questions[task][i:i + batch_size]
			embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
			all_embeddings.extend(embeddings)

		# Stack all embeddings into a single tensor
		embeddings_tensor = torch.stack(all_embeddings)
		embeddings_tensor = embeddings_tensor.cpu().numpy()  # Convert to numpy array for KMeans

		# Perform k-means clustering
		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_tensor)

		# Get predicted labels for all questions
		predicted_labels = kmeans.labels_
		return predicted_labels

def get_question_dist_via_clustering(predictions):
	# Initialize the label counter
	label_counts = Counter()
	predictions = cluster_questions(questions, task, batch_size=32, n_clusters=10)
	for prediction in predictions:
		# Update counts with the predicted labels
		label_counts.update(predicted_labels)
		
	return label_stats(label_counts)


def label_stats(label_counts):
	total = sum(label_counts.values())
	percentages = {label: (count / total * 100) for label, count in label_counts.items()}
	return percentages

if __name__ == "__main__":
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	for task_idx in range(1, 5):
		task = All_task[task_idx]
		created = load_gen_data(task)
		summary_dict = {}
		for i in range(task_idx):
			sub_task = All_task[i]
			summary_dict[sub_task] = {}
			test_data = load_orig_data(sub_task, 'train')
			predictions = cluster_questions(created, sub_task)
			label_counts_created = get_question_dist_via_clustering(predictions)
			predictions = cluster_questions(test_data, sub_task)
			label_counts_test = get_question_dist_via_clustering(predictions)
			label_counts_test = {str(k):v for k,v in label_counts_test.items()}
			label_counts_created = {str(k):v for k,v in label_counts_created.items()}
			summary_dict[sub_task]['balanced'] = label_counts_test
			summary_dict[sub_task]['unbalanced'] = label_counts_created
			print(f'for task {sub_task} the distribution of labels in synthetic data is {label_counts_created}')
			print(f'for task {sub_task} the distribution of labels in the test data is {label_counts_test}')
		with open(f"metrics/sgvqa_{task}_question_dist_via_clustering.json", 'w') as f:
			json.dump(summary_dict, f, indent=4)


