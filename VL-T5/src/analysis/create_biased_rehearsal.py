import os
import numpy as np
import pandas as pd
from tqdm import *
import torch
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from src.analysis.qtype_sim import get_embedding, QuestionTypeClassifier
desired_counts = {
	"attribute":{
		"object":{
	    0: int(5000 * 0.629),
	    1: int(5000 * 0.278),
	    2: int(5000 * 0.092),
	    },
	},
	"relation":{
		"object":{
		0: int(2500 * 0.646),
		1: int(2500 * 0.243),
		2: int(2500 * 0.110),
		},
		"attribute":{
		8: int(2500 * 0.734),
		6: int(2500 * 0.065),
		7: int(2500 * 0.026),
		1: int(2500 * 0.173),
		}
	}
}

def load_orig_data(task, split):
	np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_{split}.npy")
	data = np.load(np_path, allow_pickle=True)
	data_ = {}
	return data


def classify_questions(model, questions, task, batch_size=32):
	model.eval()  # Ensure the model is in evaluation mode
	all_predictions = []
	# Loop through batches of questions
	for i in trange(0, len(questions), batch_size):
		batch_questions = questions[i:i + batch_size]
		embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
		embeddings = torch.stack(embeddings)  # Convert list of tensors to a single tensor

		# Ensure the embeddings are on the same device as the model
		embeddings = embeddings.to(next(model.parameters()).device)
		
		with torch.no_grad():  # Disable gradient computation
			outputs = model(embeddings)  # Get the logits from the model
			probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
			predicted_labels = torch.argmax(probabilities, dim=1)  # Get the index of the max probability

		# Update counts for this batch
		# label_counts.update(predicted_labels.cpu().numpy())
		all_predictions.extend(predicted_labels.cpu().numpy())
	return all_predictions

def sample_by_predicted_labels(train_data, predictions, desired_counts):
    from collections import defaultdict
    import random

    # Mapping from labels to list of indices that have this label
    label_indices = defaultdict(list)
    for index, label in enumerate(predictions):
        label_indices[label].append(index)
    
    # Sample indices according to desired counts
    sampled_indices = []
    for label, count in desired_counts.items():
        if label in label_indices:
            if len(label_indices[label]) >= count:
                sampled_indices.extend(random.sample(label_indices[label], count))
            else:
                # If there aren't enough data as desired, take all available and print a warning
                sampled_indices.extend(label_indices[label])
                print(f"Warning: Not enough data for label {label}. Needed {count}, got {len(label_indices[label])}.")
    
    # Fetch the actual data entries corresponding to sampled indices
    sampled_data = [train_data[idx] for idx in sampled_indices]
    return sampled_data

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:  # 'wb' indicates that the file is opened for writing in binary mode
        pickle.dump(data, file)
    print(f"Data successfully saved to {filename}")


if __name__ == "__main__":

	input_dim = 768
	hidden_dim = 256
	output_dim = 20
	task = 'attribute'
	task_idx = All_task.index(task)
	classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
	new_data = {}
	count = 0
	for i in range(task_idx):
		sub_task = All_task[i]
		train_data = load_orig_data(sub_task, 'train')
		questions = [data['question'] for data in train_data]
		new_data[sub_task] = []
		if os.path.exists(f'ckpt/{sub_task}.pth'):
			print("Loading existsing checkpoint")
			ckpt = torch.load(f'ckpt/{sub_task}.pth', map_location=device)
			classifier.load_state_dict(ckpt)
		predictions = classify_questions(classifier, questions, sub_task)
		sampled_list = sample_by_predicted_labels(train_data, predictions, desired_counts[task][sub_task])
		new_data[sub_task].extend(sampled_list)
		count+=len(sampled_list)

	print(f"No. of samples present in {task} data file")
	save_to_pickle(new_data, f'{task}.pkl')




