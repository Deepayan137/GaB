import torch
import torch.nn as nn
import numpy as np
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
from src.param import parse_args
from collections import OrderedDict

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# from src.vqacl import Trainer
	import sys
	from tqdm import *
	from Question_type import All_task, Category_splits
	import os

	from src.param import parse_args

	args = parse_args()
	proj_weights = []
	savepath = f'snap/naiveblip_scratch_qtoken/'
	All_task = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory','q_causal']
	for task in All_task:
		path = os.path.join(savepath, f'{task}_LAST.pth')
		loc=None
		loc = f'cuda:{args.gpu}' if hasattr(args, 'gpu') else None
		ckpt = torch.load(path, map_location=loc)
		projection = ckpt['model']['language_projection']
		proj_weight = projection['weight']
		proj_weights.append(proj_weight.detach().cpu().numpy().flatten())

	# Convert list of weights to a numpy array
	proj_weights = np.array(proj_weights)
	
	n_samples = len(proj_weights)

	# Choose a perplexity value smaller than n_samples
	perplexity_value = min(30, 8)

	# Perform t-SNE with adjusted perplexity
	tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
	# Perform t-SNE
	print('Performing t-SNE')
	proj_weights_2d = tsne.fit_transform(proj_weights)

	# Plotting
	plt.figure(figsize=(10, 10))
	for i, label in enumerate(All_task):
		x, y = proj_weights_2d[i, :]
		plt.scatter(x, y)
		plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
		plt.savefig('metrics/cluster_head.png')
	plt.xlabel('t-SNE feature 1')
	plt.ylabel('t-SNE feature 2')
	plt.title('t-SNE Visualization of Language Projection Heads')
	plt.show()