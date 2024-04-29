import sys
import os
import json
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import *

sys.path.insert(0, '../')
from Question_type import All_task

path = f"../datasets/vqa/Partition_Q_V2_subset"
class ImageDescriptionDataset(Dataset):
	def __init__(self, task):
		with open(os.path.join(path, f'{task}.json')) as f:
			self.data = json.load(f)
		self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
		data_dir = "../datasets/COCO"
		self.source_dir = {
			'train2014': os.path.join(data_dir, f'train2014'),
			'val2014': os.path.join(data_dir, f'val2014'),
		}
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		datum = self.data[idx]
		img_id = datum['img_id']
		source = img_id.split('_')[1]
		f = f"{os.path.join(self.source_dir[source], img_id)}.jpg"
		if os.path.exists(f):
			image = Image.open(f).convert("RGB")
		else:
			raise "image path does not exists"

		# prompt = """<image>\nUSER: Provide a detailed description of this image, including the count of items, their colors (such as their clothes or primary objects), what they're doing (actions), and their spatial positions (such as 'left', 'right', 'front', 'back', etc.).\nASSISTANT:"""
		prompt = """<image>\nUSER:Analyze the image and generate relevant question-answer pairs that focus ONLY on the locations. Questions should begin with 'Where is the' or 'what room is'. Example question answer pairs: 'what room is this?kitchen<sep>where is the dog?beach'"""
		inputs = self.processor(text=prompt, images=image, return_tensors="pt")
		return inputs


class LlavaCaption(object):
	def __init__(self):
		self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
		self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
		self.device = "cuda"
		self.model.to(self.device)

	def generate_caption(self, batch):
		device = next(self.model.parameters()).device
		
		input_ids = batch['input_ids'].squeeze().to(device)
		attention_mask = batch['attention_mask'].squeeze().to(device)
		pixel_values = batch['pixel_values'].squeeze().to(device)
		generate_ids = self.model.generate(pixel_values=pixel_values, 
			attention_mask=attention_mask, 
			input_ids=input_ids, 
			max_length=512)
		out = self.processor.batch_decode(
			generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
			)
		import pdb;pdb.set_trace()
		return out

if __name__ == "__main__":
	model = LlavaCaption()
	for task_idx, task in enumerate(All_task):
		if task_idx > 0:
			dataset=ImageDescriptionDataset(task)
			loader = DataLoader(dataset, batch_size=8)
			for batch in loader:
				model.generate_caption(batch)
				





