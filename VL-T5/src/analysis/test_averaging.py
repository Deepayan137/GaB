import torch
import torch.nn as nn
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
from src.param import parse_args
from collections import OrderedDict
if __name__ == "__main__":
	
	# from src.vqacl import Trainer
	import sys
	from tqdm import *
	from Question_type import All_task, Category_splits
	import os

	from src.param import parse_args

	args = parse_args()
	task1 = 'q_action'
	task2 = 'q_subcategory'
	path1 = f'snap/naiveblip_scratch_qtoken/{task1}_LAST.pth'
	path2 = f'snap/naiveblip_scratch_qtoken/{task2}_LAST.pth'
	
	loc=None
	if loc is None and hasattr(args, 'gpu'):
		loc = f'cuda:{args.gpu}'
	ckpt1 = torch.load(path1, map_location=loc)
	ckpt2 = torch.load(path2, map_location=loc)

	# Corrected loading of qtokens2
	qtokens1 = ckpt1['model']['query_tokens']
	qtokens2 = ckpt2['model']['query_tokens']

	projection1 = ckpt1['model']['language_projection']
	projection2 = ckpt2['model']['language_projection']

	# Define the interpolation parameter alpha (between 0 and 1)
	alpha = 0.5

	# Interpolate weights for query tokens and language projection
	print("Interpolating...")
	interpolated_qtokens = (1 - alpha) * qtokens1 + alpha * qtokens2
	# Interpolate weights for language projection
	interpolated_projection_weight = (1 - alpha) * projection1['weight'] + alpha * projection2['weight']
	interpolated_projection_bias = (1 - alpha) * projection1['bias'] + alpha * projection2['bias']
	interpolated_projection = OrderedDict([
		('weight', interpolated_projection_weight),
		('bias', interpolated_projection_bias)
	])

	state_dict_to_save = {
		"optimizer": ckpt1["optimizer"],
		"examplar": ckpt1["examplar"],  # Verify the correct key for the examplar set
		"model": {
			"query_tokens": interpolated_qtokens,
			"language_projection": interpolated_projection,
			# Include other model parameters as needed
		}
	}

	# import pdb;pdb.set_trace()
	# Ensure the save directory exists or handle its creation
	save_path = 'snap/naiveblip_scratch_qtoken/q_action_subcategory_LAST.pth'
	torch.save(state_dict_to_save, save_path)
	print("Average weights saved")

	# backbone = 'Salesforce/blip2-opt-2.7b'
	# config =Blip2Config.from_pretrained(backbone)
	# processor = AutoProcessor.from_pretrained(backbone)
	# model = Blip2ForConditionalGeneration.from_pretrained(backbone, config=config)
	# device = 'cuda'
	# model = model.to(device)
	# checkpoint = torch.load(save_path, map_location=loc)

	# model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
	# model.language_projection.load_state_dict(checkpoint['model']['language_projection'])