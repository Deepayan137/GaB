import os
import numpy as np
import pandas as pd
from tqdm import *
import torch
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier
from src.analysis.question_distribution import classify_questions, sample_by_predicted_labels
import sys
sys.insert.path(0, '../')
from Question_type import cat_dict
desired_counts = {
    "attribute": {
        "object": {
            0: int(5000 * 0.693),
            1: int(5000 * 0.288),
            2: int(5000 * 0.017),
        },
    },
    "relation": {
        "object": {
            0: int(2500 * 0.428),
            1: int(2500 * 0.562),
            2: int(2500 * 0.009),
        },
        "attribute": {
            0: int(2500 * 0.025),
            5: int(2500 * 0.345),
            4: int(2500 * 0.062),
            8: int(2500 * 0.337),
            6: int(2500 * 0.062),
            7: int(2500 * 0.018),
            1: int(2500 * 0.157),
        }
    },
    "logical": {
        "object": {
            0: int(1666 * 0.289),
            1: int(1666 * 0.626),
            2: int(1666 * 0.084),
        },
        "attribute": {
            0: int(1666 * 0.106),
            5: int(1666 * 0.223),
            4: int(1666 * 0.048),
            8: int(1666 * 0.281),
            6: int(1666 * 0.062),
            7: int(1666 * 0.003),
            1: int(1666 * 0.244),
        },
        "relation": {
            0: int(1666 * 0.409),
            1: int(1666 * 0.086),
            5: int(1666 * 0.409),
            7: int(1666 * 0.081),
            13: int(1666 * 0.014)
        }
    },
    "knowledge": {
        "object": {
            0: int(1250 * 0.196),
            1: int(1250 * 0.541),
            2: int(1250 * 0.262),
        },
        "attribute": {
            0: int(1250 * 0.162),
            5: int(1250 * 0.146),
            4: int(1250 * 0.043),
            8: int(1250 * 0.224),
            6: int(1250 * 0.031),
            7: int(1250 * 0.068),
            1: int(1250 * 0.32),
        },
        "relation": {
            0: int(1250 * 0.318),
            1: int(1250 * 0.259),
            5: int(1250 * 0.330),
            7: int(1250 * 0.072),
            13: int(1250 * 0.023),
        },
        "logical": {
            2: int(1250 * 0.71),
            6: int(1250 * 0.189),
            10: int(1250 * 0.043),
            3: int(1250 * 0.044),
            0: int(1250 * 0.012)
        }
    }
}


def load_orig_data(task, split):
	np_path = os.path.join('../datasets/npy', 'function', f"fcl_mmf_{task}_{split}.npy")
	data = np.load(np_path, allow_pickle=True)
	data_ = {}
	return data


def save_to_pickle(data, filename):
	with open(filename, 'wb') as file:  # 'wb' indicates that the file is opened for writing in binary mode
		pickle.dump(data, file)
	print(f"Data successfully saved to {filename}")


if __name__ == "__main__":

	input_dim = 768
	hidden_dim = 256
	task = 'attribute'
	task_idx = All_task.index(task)
	
	
	for task_idx in range(1, 2):
		task = All_task[task_idx]
		new_data = {}
		count = 0
		split=5000/task_idx
		for i in range(task_idx):
			sub_task = All_task[i]
			train_data = load_orig_data(sub_task, 'train')
			questions = [data['question'] for data in train_data]
			new_data[sub_task] = []
			output_dim = len(cat_dict[sub_task])
			classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
			if os.path.exists(f'ckpt/{sub_task}.pth'):
				print("Loading existsing checkpoint")
				ckpt = torch.load(f'ckpt/{sub_task}.pth', map_location=device)
				classifier.load_state_dict(ckpt)
			predictions = classify_questions(classifier, questions, sub_task)
			sampled_list = sample_by_predicted_labels(train_data, predictions, desired_counts[task][sub_task], total_target=split)
			new_data[sub_task].extend(sampled_list)
			count += len(sampled_list)
		print(f"No. of samples present in {task} data file: {count}")
		save_to_pickle(new_data, f'../datasets/npy_biased/{task}.pkl')




