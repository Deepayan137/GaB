import json
import random
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


def inference_qa(image_path, question, max_new_tokens=2):
	image = Image.open(image_path).convert("RGB")
	inputs = processor(image, text=question, 
		truncation=True, return_tensors="pt").to(device)
	# target_ids = processor.tokenizer.encode(answer, max_length=10, truncation=True)
	pixel_values = inputs["pixel_values"].to(device)
	input_ids = inputs["input_ids"].to(device)
	# target_ids = target_ids.to(device)
	output = model.generate(**inputs, max_new_tokens=20
	    # max_new_tokens=max_new_tokens,repetition_penalty=1, num_beams=5, length_penalty=-1
	    )
	pred_ans =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)
	return pred_ans

def inference_cap(image_path, prompt=None, max_new_tokens=20):
	image = Image.open(image_path).convert("RGB")
	if not prompt:
		inputs = processor(image, return_tensors="pt").to(device)
	else:
		inputs = processor(image, text=prompt, return_tensors="pt").to(device)
	generated_ids = model.generate(**inputs, max_new_tokens=20)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
	return generated_text

def gen_cap_loop():
	cap_dict = {}
	for task in tqdm(All_task):
		path = f"../datasets/vqa/Partition_Q_V2/karpathy_train_{task}.json"
		
		f = open(path, 'r')
		data = json.load(f)
		subset = random.sample(data, 10)
		cap_dict[task] = []
		for item in subset:
			img_name = item['img_id']+'.jpg'
			if 'train2014' in img_name:
				split="train"
			else:
				split = 'val'
			img_dir = f"../datasets/COCO/{split}2014/"
			img_path = os.path.join(img_dir, img_name)
			cap = inference_cap(img_path)
			import pdb;pdb.set_trace()
			cap_dict[task].append((img_name, cap))
	with open('test.json', 'w')	as f:
		json.dump(cap_dict, f, indent=4)
if __name__ == "__main__":
	# gen_cap_loop()
	# data_dir = "/nfs/data_todi/datasets/COCO2014/val2014"
	# data_dir = "/home/deepayan.das/projects/VQACL/datasets/COCO/val2014/"
	data_dir = '/home/deepayan.das/projects/SG-CLVQA/datasets/gvqa//'
	image_name = "2343078.jpg"
	image_path = os.path.join(data_dir, image_name)
	question = "Who wears the jacket?"
	sent = f"Question:{question} Answer:"
	max_new_tokens = 20
	# pred_ans = inference_qa(image_path, sent, max_new_tokens)
	# print(pred_ans)
	gen_cap_loop()
	import pdb;pdb.set_trace()
	# context = [
 #   		("What is the image of?", "a bedroom"),
 #   		("Are there frames on the wall?", "Yes"),
	# ]
	# question = "what number of frames are on the wall?"
	# template = "Question: {} Answer: {}."

	# prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
	# cap = inference_cap(image_path, prompt="Describe this image in detail.")
	# print(cap)
	# import pdb;pdb.set_trace()
 
