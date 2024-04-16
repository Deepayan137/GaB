import os
import json
import sys
import random
import shutil
sys.path.insert(0, '../')
from Question_type import All_task
from src.analysis.test_context import inference_cap
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import *

class ImageDescriptionDataset(Dataset):
	def __init__(self, image_paths):
		self.image_paths = image_paths
		self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx][0]
		qa = f"Q:{self.image_paths[idx][1]}, A:{self.image_paths[idx][2]}"
		image_id = os.path.basename(image_path).split('.')[0]
		image = Image.open(image_path).convert("RGB")
		prompt = """<image>\nUSER: Provide a detailed description of this image, including the count of items, their colors (such as their clothes or primary objects), what they're doing (actions), and their spatial positions (such as 'left', 'right', 'front', 'back', etc.).\nASSISTANT:"""
		# inputs = self.processor(text=prompt, images=image, return_tensors="pt")
		# return inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'], image_id
		return image_id, image, prompt, qa

def collate_fn(batch):
	id_ = [item[0] for item in batch]
	images = [item[1] for item in batch]
	prompts = [item[2] for item in batch]
	qas = [item[3] for item in batch]
	return {
		"id":id_, 
		"images":images, 
		"prompts": prompts,
		"qas": qas
	}
class LlavaCaption(object):
	def __init__(self):
		self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
		self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
		self.device = "cuda"
		self.model.to(self.device)

	def generate_caption(self, batch):
		images = batch["images"]
		prompts = batch["prompts"]
		inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(self.device)
		generate_ids = self.model.generate(**inputs, max_length=512)
		out = self.processor.batch_decode(
			generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
			)
		
		return out



if __name__ == "__main__":
	cap_dict = {}
	path = f"../datasets/vqa/Partition_Q_V2/"
	dest_root = f"../datasets/vqa/Partition_Q_V2_subset"
	root = f"../datasets/COCO/"
	os.makedirs(dest_root, exist_ok=True)
	# os.makedirs(img_dir, exist_ok=True)
	# All_task = ['q_location']
	img_paths = []
	for task in All_task:
		fname = f"karpathy_train_{task}.json"
		source = os.path.join(path, fname)
		dest = os.path.join(dest_root, fname)
		if not os.path.exists(dest):
			f = open(source, 'r')
			data = json.load(f)
			data_subset = random.sample(data, 5)
			with open(dest, 'w')	as f:
				json.dump(data_subset, f)

		else:
			with open(dest, 'r') as f:
				data_subset = json.load(f)

		for data in data_subset:
			img_id = data['img_id']
			img_folder = img_id.split('_')[1]
			img_path = f"{os.path.join(root, img_folder, img_id)}.jpg"
			ans = max(data['label'], key=lambda k: data['label'][k])
			ques = data['sent']
			if img_path not in img_paths:
				img_paths.append((img_path, ques, ans))

	print(f"Number of Images:{len(img_paths)}")
	dataset = ImageDescriptionDataset(img_paths)
	dataloader = DataLoader(
		dataset, 
		batch_size=8, 
		num_workers=0, 
		collate_fn=collate_fn)	
	model = LlavaCaption()
	id_cap_pairs = []
	for batch in tqdm(dataloader):
		captions = model.generate_caption(batch)
		id_cap_pairs.extend(list(zip(batch['id'], captions, batch['qas'])))
	print(f"Number of captions are:{len(id_cap_pairs)}")
	cap_dict = {}
	for id_, caption, qas in id_cap_pairs:
		cap_dict[id_] = [caption.split("ASSISTANT:")[1].strip(), qas]
		

	with open("../datasets/captions_COCO_subset.json", "w") as f:
		json.dump(cap_dict, f, indent=4)