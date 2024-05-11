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
from tqdm import *
from Question_type import *
import os
import random

from src.param import parse_args
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
	for task in All_task:
		source_file = '../datasets/vqa/Partition_Q_V2/karpathy_test_' + f'{task}.json'
		dest_file = os.path.join(dest_root, f'karpathy_test_{task}.json')
		if not os.path.exists(dest_file):
			shutil.copyfile(source_file, dest_file)
		else:
			print("File already present, no need to copy")
	print("Complete...")


def create_rehearsal_memory(dest_root):
	for task_idx in range(len(All_task)):
		Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
		os.makedirs(dest_root, exist_ok=True)
		if task_idx > 0:
			each_memory = 5000
			data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{All_task[task_idx]}.json')
			with open(data_info_path, 'r') as f:
				data_info_dicts = json.load(f)
			random.shuffle(data_info_dicts)
			each_memory_for_cate = int(each_memory / len(Category_splits))
			for cate in Category_splits:
				num = 0
				for _d in data_info_dicts:
					img_id = _d['img_id']
					if img_id in ImgId_cate_map:
						if ImgId_cate_map[img_id] in Category_splits[cate]:
							Examplar_set[cate].append(_d)
							num += 1
							if num >= each_memory_for_cate:
								break
			
			All_examplar = []
			for cate in Category_splits:
				All_examplar.extend(Examplar_set[cate])
			print(f"Data present in {All_task[task_idx]} split: {len(All_examplar)}")
			dest = os.path.join(dest_root, f'karpathy_train_{All_task[task_idx]}.json')
			with open(dest, 'w') as f:
				json.dump(All_examplar, f, indent=4)

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

def inference_cap(images, prompts=None, temp=0.7):
	inputs = processor(images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
	generated_ids = model.generate(**inputs, max_new_tokens=56, num_beams=5, temperature=temp, do_sample=True)
	captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
	if prompts:
		empty_indices = [index for index, cap in enumerate(captions) if cap.strip() == ""]
		prompts = [prompt for index, prompt in enumerate(prompts) if index not in empty_indices]
		captions = [cap for index, cap in enumerate(captions) if index not in empty_indices]
		captions = [f"{prompt} {cap.strip()}" for prompt, cap in zip(prompts, captions)]
		return captions
	return captions[0].strip()



spec_prompt_sing = ["the {noun} in the image is", "the {noun} in the image is doing", "the {noun} in the image is wearing a", "the {noun} in the image seems", ]
spec_prompt_mult = ["the {noun} in the image are", "the {noun} in the image are doing", "the {noun} in the image are wearing", "the {noun} in the image seem",]

def generate_prompts(features):
	prompts = ["The image is of", "The place shown in the image is", "The color of", "the image is set in", "the image is taken during", "In the background is", 'On the right of']
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
	path = f"../datasets/npy/"
	dest_root = f"../datasets/vqa/npy_cap"
	os.makedirs(dest_root, exist_ok=True)
	copy_val_files(path, dest_root)
	# create_rehearsal_memory(dest_root)
	model_name = "Salesforce/blip2-opt-2.7b"
	model, processor = get_model(model_name)

	img_paths = []
	task2id = {All_task[i]:i for i in range(len(All_task))}
	id2task = {v:k for k, v in task2id.items()}
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) 
	task = All_task[task_idx]
	# task_idx = 0
	if task_idx > -1:
		fname = f"fcl_mmf_{task}_train.json"
		source = os.path.join(path, fname)
		dest = os.path.join(dest_root, fname)
		
		data_subset = np.load()
		random.shuffle(data_subset)
		new_data = []
		count=0
		for _d in tqdm(data_subset):
			img_name = f"{_d['img_id']}.jpg"
			split = "train" if 'train2014' in img_name else 'val'
			img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
			image = Image.open(img_path).convert("RGB")
			initial_caption = inference_cap(image, temp=2.5)
			cap_list = []
			# cap_dict[img_name]["captions"] = [initial_caption]
			cap_list.append(initial_caption)
			features = extract_features(initial_caption)
			prompts = generate_prompts(features)
			num_prompts = len(prompts)
			images = num_prompts * [image]
			caps = inference_cap(images, prompts,temp=1.8)
			cap_list.extend(caps)
			caption = '. '.join([item.capitalize() for item in cap_list])
			# cap_dict[img_name]["captions"].extend(caps)
			_d['caption'] = caption
			new_data.append(_d)
			count+=1
		with open(dest, 'w') as f:
			json.dump(new_data, f, indent=4)
	print(f"Finished. Captions printed{count}")
