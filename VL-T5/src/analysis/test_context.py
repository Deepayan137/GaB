import torch
import torch.nn as nn
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
# from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args
from collections import OrderedDict

import sys
from tqdm import *
from Question_type import All_task, Category_splits
import os

from src.param import parse_args
args = parse_args()
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
device = 'cuda'


# path = f'snap/naiveblip_scratch_qtoken/q_location_LAST.pth'
model_name = "Salesforce/blip2-opt-2.7b"

config = Blip2Config.from_pretrained(model_name)
model = NaiveBLIP2.from_pretrained(model_name, config=config)
model.to(device)
# loc=None
# if loc is None and hasattr(args, 'gpu'):
# 	loc = f'cuda:{args.gpu}'
# ckpt = torch.load(path, map_location=None)

# Corrected loading of qtokens2
# qtokens = ckpt['model']['query_tokens']

# projection = ckpt['model']['language_projection']
# model.query_tokens.data.copy_(ckpt['model']['query_tokens'])
# model.language_projection.load_state_dict(ckpt['model']['language_projection'])


def inference(image_path, question, max_new_tokens=2):
	image = Image.open(image_path).convert("RGB")
	inputs = processor(image, text=question, max_length=20, 
		truncation=True, return_tensors="pt")
	# target_ids = processor.tokenizer.encode(answer, max_length=10, truncation=True)
	pixel_values = inputs["pixel_values"].to(device)
	input_ids = inputs["input_ids"].to(device)
	# target_ids = target_ids.to(device)
	output = model.generate(input_ids=input_ids,pixel_values=pixel_values,
	    max_new_tokens=max_new_tokens,repetition_penalty=1, num_beams=5, length_penalty=-1)
	pred_ans =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)
	return pred_ans

if __name__ == "__main__":
	data_dir = "/nfs/data_todi/datasets/COCO2014/val2014"
	# data_dir = "."
	image_name = "COCO_val2014_000000263828.jpg"
	image_path = os.path.join(data_dir, image_name)
	question = "what material is the seat of the bike made out of?"
	sent = f"Question:{question} Answer:"
	max_new_tokens = 20
	pred_ans = inference(image_path, sent, max_new_tokens)
	print(pred_ans)
	import pdb;pdb.set_trace()
 
