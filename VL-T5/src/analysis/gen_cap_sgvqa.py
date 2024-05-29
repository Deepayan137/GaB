import json
import random
import spacy
import torch
import torch.nn as nn
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset
from transformers import AutoProcessor
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
from PIL import Image
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args
from collections import OrderedDict
import shutil
import sys
import numpy as np
from tqdm import *
from Question_type import *
import os
import random

from src.param import parse_args

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nlp = spacy.load("en_core_web_sm")
device = 'cuda'
args = parse_args()

def get_model(model_name):
	processor = AutoProcessor.from_pretrained(model_name)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	config = Blip2Config.from_pretrained(model_name)
	model = Blip2ForConditionalGeneration.from_pretrained(model_name, config=config)
	model.to(device)
	# if os.path.exists(ckpt_path):
	# 	checkpoint = torch.load(ckpt_path, map_location=device)
	# 	model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
	# 	model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection'])
	return model, processor

def copy_val_files(source, dest_root):
	for task in Sg_task['function']['oarlks']:
		source_file = os.path.join(source, f"fcl_mmf_{task}_val.npy")
		dest_file = os.path.join(dest_root, f"fcl_mmf_{task}_val.npy")
		if not os.path.exists(dest_file):
			shutil.copyfile(source_file, dest_file)
		else:
			print("File already present, no need to copy")
	print("Complete...")


def create_rehearsal_memory(source, dest_root, tasks):
	for task_idx in range(len(tasks)):
		os.makedirs(dest_root, exist_ok=True)
		if task_idx > -1:
			each_memory = 5000
			fname = f"fcl_mmf_{tasks[task_idx]}_train.npy"
			dest = os.path.join(dest_root, fname.split('.')[0]+'.json')
			if not os.path.exists(dest):
				source_file = os.path.join(source, fname)
				data_info_dicts = np.load(source_file, allow_pickle=True)
				random.shuffle(data_info_dicts)
				data_subset = []
				num=0
				for _d in data_info_dicts:
					data_subset.append(_d)
					num += 1
					if num >= each_memory:
						break
				
				with open(dest, 'w') as f:
					json.dump(data_subset, f, indent=4)
			else:
				print("File already present")

def inference_qa(image_path, max_new_tokens=20):
	image = Image.open(image_path).convert("RGB")
	inputs = processor(image, text=question, 
		truncation=True, return_tensors="pt").to(device)
	# target_ids = processor.tokenizer.encode(answer, max_length=10, truncation=True)
	pixel_values = inputs["pixel_values"].to(device)
	input_ids = inputs["input_ids"].to(device)
	# target_ids = target_ids.to(device)
	output = model.get_questions(inputs, max_new_tokens=50, num_beams=5, num_return_sequences=3)
	
	pred_question =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)
	return pred_question

def inference_cap(images, prompts=None, temp=0.7, max_new_tokens=32):
	inputs = processor(images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
	generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=5, temperature=temp, do_sample=True, repetition_penalty=1.2)
	captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
	if prompts:
		empty_indices = [index for index, cap in enumerate(captions) if cap.strip() == ""]
		prompts = [prompt for index, prompt in enumerate(prompts) if index not in empty_indices]
		captions = [cap for index, cap in enumerate(captions) if index not in empty_indices]
		captions = [f"{prompt} {cap.strip()}" for prompt, cap in zip(prompts, captions)]
		return captions
	return captions[0].strip()



spec_prompt_sing = ["The {noun} in the image is", "The {noun} in the image is doing", "The {noun} in the image is wearing a", "The {noun} in the image seems", ]
spec_prompt_mult = ["The {noun} in the image are", "The {noun} in the image are doing", "The {noun} in the image are wearing", "The {noun} in the image seem",]

def generate_prompts(features):
	prompts = ["The place shown in the image is", "The color of", 'On the right of', 'On the left of', 'The weather is']
	for noun, number in features['nouns']:
		if number == 'plural':
			if noun in ['man', 'woman', 'father', 'daughter','boy', 'girl', 'person', 'child', 'baby', 'animal', 'dog', 'cat', 'zebra', 'giraffe']:
				for prompt in spec_prompt_mult:
					prompts.append(prompt.format(noun=noun))
			
		else:
			if noun in ['men', 'women', 'boys', 'girls', 'persons', 'children', 'babies', 'animals', 'dogs', 'cats', 'zebras', 'giraffes']:
				for prompt in spec_prompt_sing:
					prompts.append(prompt.format(noun=noun))
	return prompts

def extract_features(sentence):
	doc = nlp(sentence)
	features = {
        'nouns': [],
    }
	for token in doc:
		if token.pos_ == "NOUN":
			number = "plural" if token.tag_ == "NNS" else "singular"
			features["nouns"].append((token.text, number))
	return features
				


if __name__ == "__main__":
	# import argparse
	# parser = argparse.ArgumentParser(description="Process some integers.")
	# parser.add_argument('--split', dest='split', default='train', type=str,
 #                        help='Disable the splitting feature (default is to enable splitting)')
	# pargs = parser.parse_args()
	source = f"../datasets/npy/function"
	dest_root = f"../datasets/npy_cap/function"
	os.makedirs(dest_root, exist_ok=True)
	# copy_val_files(source, dest_root)
	print("Creating rehearsal")
	tasks = Sg_task['function']['oarlks']
	# create_rehearsal_memory(source, dest_root, tasks)
	model_name = "Salesforce/blip2-opt-2.7b"
	model, processor = get_model(model_name)

	img_paths = []
	task2id = {All_task[i]:i for i in range(len(All_task))}
	id2task = {v:k for k, v in task2id.items()}
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) 
	task_idx = 2
	task = tasks[task_idx]
	split = 'train'
	# task_idx = 0
	if task_idx > -1:
		fname = f"fcl_mmf_{task}_{split}"
		if 'train' in fname:
			fname = fname +'.json'
			dest = os.path.join(dest_root, fname)
			with open(dest, 'r') as f:
				data_subset = json.load(f)
		elif 'val' in fname:
			fname = fname +'.npy'
			dest = os.path.join(dest_root, fname)
			data_subset = np.load(dest, allow_pickle=True)
		print(f"Loaded {dest}")
		new_data = []
		count=0
		for _d in tqdm(data_subset):
			img_name = f"{_d['image_id']}.jpg"
			if task_idx != 5:
				image_dir = 'gvqa'
			else:
				image_dir = f"textvqa_train"
			img_path = os.path.join(f"../datasets/{image_dir}", img_name)
			image = Image.open(img_path).convert("RGB")
			import pdb;pdb.set_trace()
			initial_caption = inference_cap(image, temp=2.5, max_new_tokens=48)
			# cap_list = []
			# # cap_dict[img_name]["captions"] = [initial_caption]
			# cap_list.append(initial_caption)
			# features = extract_features(initial_caption)
			# prompts = generate_prompts(features)
			# num_prompts = len(prompts)
			# images = num_prompts * [image]
			# caps = inference_cap(images, prompts,temp=1.8, max_new_tokens=32)
			# cap_list.extend(caps)
			# caption = '. '.join([item.capitalize() for item in cap_list])
			# cap_dict[img_name]["captions"].extend(caps)
			_d['caption'] = initial_caption
			new_data.append(_d)
			count+=1
			# if count >= 5000:
			# 	break
		with open(os.path.join(dest_root, fname.split('.')[0]+'.json'), 'w') as f:
			json.dump(new_data, f, indent=4)
	print(f"Finished. Captions printed {count}")