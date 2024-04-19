import json
import random
import spacy
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

nlp = spacy.load("en_core_web_sm")

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
	output = model.generate(**inputs, max_new_tokens=50, num_beams=5, num_return_sequences=3)
	
	pred_ans =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)
	return pred_ans

def inference_cap(images, prompts=None, temp=0.7):
	inputs = processor(images, text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=20).to(device)
	generated_ids = model.generate(**inputs, max_new_tokens=50, num_beams=5, temperature=temp, do_sample=True)
	captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
	if prompts:
		empty_indices = [index for index, cap in enumerate(captions) if cap.strip() == ""]
		prompts = [prompt for index, prompt in enumerate(prompts) if index not in empty_indices]
		captions = [cap for index, cap in enumerate(captions) if index not in empty_indices]
		captions = [f"{prompt} {cap.strip()}" for prompt, cap in zip(prompts, captions)]
		return captions
	return captions[0].strip()



spec_prompt_sing = ["the {noun} in the image is", "the {noun} in the image is doing", "the {noun} in the image is wearing a", "the {noun} in the image seems"]
spec_prompt_mult = ["the {noun} in the image is", "the {noun} in the image are doing", "the {noun} in the image are wearing", "the {noun} in the image seem"]

def generate_prompts(features):
	prompts = ["the image is of", "the image is set in", "the image is taken during", "the main object in the image is", "on the background is", "on the foreground is"]
	for noun, number in features['nouns']:
		if noun in ['man', 'woman', 'father', 'daughter','boy', 'girl', 'men', 'women', 'persons', 'person', 'people', 'child', 'baby', 'animal', 'dog', 'cat','animals']:
			if number == 'plural':
				for prompt in spec_prompt_mult:
					prompts.append(prompt.format(noun=noun))
			else:
				for prompt in spec_prompt_sing:
					prompts.append(prompt.format(noun=noun))
	return prompts

def extract_features(sentence):
	doc = nlp(sentence)
	features = {
        'nouns': [],
        # 'adjectives': [],
        # 'verbs': []
    }
	for token in doc:
		if token.pos_ == "NOUN":
			number = "plural" if token.tag_ == "NNS" else "singular"
			features["nouns"].append((token.text, number))
		# elif token.pos_ == 'ADJ':
		# 	features['adjectives'].append(token.text)
		# elif token.pos_ == 'VERB':
		# 	features['verbs'].append(token.text)
	return features
				
def gen_cap_loop():
	cap_dict = {}
	for i, task in (enumerate(All_task)):
		if i > 0:
			num = 5000//i
			print(f"Now is task {All_task[i]} and number of samples is {num}")
			path = os.path.join("../datasets/vqa/Partition_Q_V2", f"karpathy_train_{All_task[i]}.json")
			with open(path, 'r') as f:
				data = json.load(f)
			subset = random.sample(data, num)
			
			cap_dict = {}

			for item in tqdm(subset):
				img_name = f"{item['img_id']}.jpg"
				split = "train" if 'train2014' in img_name else 'val'
				img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
				initial_caption = inference_cap(img_path, temp=2.5)
				cap_dict[img_name] = {}
				cap_dict[img_name]["captions"] = [initial_caption]
				features = extract_features(initial_caption)
				prompts = generate_prompts(features)
				import pdb;pdb.set_trace()
				for prompt in prompts:
					cap = inference_cap(img_path, prompt)
					cap_dict[img_name]["captions"].extend(cap)
			
			with open(f'../datasets/vqa/captions/{task}.json', 'w') as f:
				json.dump(cap_dict, f, indent=4)
if __name__ == "__main__":
	
	gen_cap_loop()
	# data_dir = "/nfs/data_todi/datasets/COCO2014/val2014"
	# data_dir = "/home/deepayan.das/projects/VQACL/datasets/COCO/val2014/"
	# data_dir = '/home/deepayan.das/projects/SG-CLVQA/datasets/gvqa//'
	# image_name = "2343078.jpg"
	# image_path = os.path.join(data_dir, image_name)
	# question = "Who wears the jacket?"
	# sent = f"Question:{question} Answer:"
	# max_new_tokens = 20
	# pred_ans = inference_qa(image_path, sent, max_new_tokens)
	# print(pred_ans)
	
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
 
