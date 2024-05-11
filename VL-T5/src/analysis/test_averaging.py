import torch
import torch.nn as nn
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
from src.param import parse_args
from collections import OrderedDict
if __name__ == "__main__":
	
	# from src.vqacl import Trainer
	import sys
	from tqdm import *
	from Question_type import All_task, Category_splits, Sg_task
	import os

	from src.param import parse_args

	args = parse_args()
	all_tasks = Sg_task['function']['oarlks']
	path1 = f'snap/blip_base/base.pth'
	loc= "cuda" if torch.cuda.is_available() else "cpu"
	ckpt1 = torch.load(path1, map_location=loc)
	qtokens1 = ckpt1['model']['query_tokens']
	projection1 = ckpt1['model']['language_projection']
	# for task in all_tasks:
	path2 = f'snap/naiveblip_sgvqa_mem_avg_last/object_LAST.pth'
	ckpt2 = torch.load(path2, map_location=loc)
	qtokens2 = ckpt2['model']['query_tokens']
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
		"optimizer": ckpt2["optimizer"],
		"examplar": ckpt2["examplar"],  # Verify the correct key for the examplar set
		"model": {
			"query_tokens": interpolated_qtokens,
			"language_projection_answers": interpolated_projection,
			"language_projection_questions":interpolated_projection
			# Include other model parameters as needed
		}
		}

	# import pdb;pdb.set_trace()
	# Ensure the save directory exists or handle its creation
	save_path = f'snap/naiveblip_sgvqa_mem_avg_last/object_AVG_LAST.pth'
	torch.save(state_dict_to_save, save_path)
	print(f"Average weights saved @ {save_path}")

	# backbone = 'Salesforce/blip2-opt-2.7b'
	# config =Blip2Config.from_pretrained(backbone)
	# processor = AutoProcessor.from_pretrained(backbone)
	# model = Blip2ForConditionalGeneration.from_pretrained(backbone, config=config)
	# device = 'cuda'
	# model = model.to(device)
	# checkpoint = torch.load(save_path, map_location=loc)

	# model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
	# model.language_projection.load_state_dict(checkpoint['model']['language_projection'])