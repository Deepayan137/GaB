import json
import os
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Config, AutoProcessor, Blip2ForConditionalGeneration
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args
from src.utils import LossMeter
import random


class QuestionDataset(Dataset):
	def __init__(self, task='object', split='train', scenario='function', verbose=True, args=None):
		super().__init__()
		filename = f'fcl_mmf_{task}_{split}.json'
		print(f"Now is task {task}. Loading data for it")
		data_path = os.path.join('../datasets/npy_cap_all', scenario, filename)
		with open(data_path, 'r') as f:
			data = json.load(f)
		self.data = data
		self.split = split
		self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		self.processor.tokenizer.padding_side = 'right'
		self.processor.tokenizer.truncation_side = 'right'

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		out_dict = {}
		datum = self.data[idx]
		question = datum['question']
		caption = datum['caption']
		img_id = datum['image_id']
		image_dir = 'gvqa'
		f = f"{os.path.join(f'../datasets/{image_dir}', img_id)}.jpg"
		if os.path.exists(f):
			image = Image.open(f).convert("RGB")
		else:
			raise "image path does not exists"
		if 'answer' in datum:
			if isinstance(datum['answer'], list):
				answer = datum['answer'][0]
			else:
				answer = datum['answer']
		max_length = 60
		sent = f"Caption:{caption} Answer:{answer}. Question:"
		inputs = self.processor(image, text=sent, max_length=max_length, 
			truncation=True, return_tensors="pt")
		out_dict['pixel_values'] = inputs['pixel_values']
		out_dict['input_ids'] = inputs['input_ids']
		out_dict['input_length'] = len(inputs["input_ids"][0])
		out_dict['sent'] = sent
		target_ids = self.processor.tokenizer.encode(question, max_length=10, truncation=True)
		out_dict['target'] = question
		out_dict['target_ids'] = torch.LongTensor(target_ids)
		out_dict['target_length'] = len(target_ids)
		return out_dict

	def collate_fn(self, batch):
		batch_entry = {}
		B = len(batch)
		S_W_L = max(entry['input_length'] for entry in batch)
		input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		vis_feats = torch.zeros(B, 3, 224, 224, dtype=torch.float)
		if 'target_ids' in batch[0]:
			T_W_L = max(entry['target_length'] for entry in batch)
			target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		for i, entry in enumerate(batch):
			input_ids[i, :entry['input_length']] = entry['input_ids'][0]
			vis_feats[i] += entry['pixel_values'][0]
			target_ids[i, :entry['target_length']] = entry['target_ids']
		batch_entry['input_ids'] = input_ids
		batch_entry['pixel_values'] = vis_feats
		batch_entry['target_ids'] = target_ids
		return batch_entry

class QuestionTrainer():
	def __init__(self, args, task):
		model_name = "Salesforce/blip2-opt-2.7b"
		self.task = task
		self.args = args
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.processor = AutoProcessor.from_pretrained(model_name)
		config = Blip2Config.from_pretrained(model_name)
		self.config = config
		self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, config=config)
		for name, param in self.model.vision_model.named_parameters():
			param.requires_grad = False
		print("Freeze vision encoder")
		self.model.vision_model = self.model.vision_model.eval()
		num_layers = len(self.model.qformer.encoder.layer)
		# Freeze all parameters of the query transformer by default
		print("Freezing Qformer parameters")
		for param in self.model.qformer.parameters():
			param.requires_grad = False
		print('Unfreezing lang proj head')
		self.model.query_tokens.requires_grad = True
		self.model.language_projection.requires_grad = True
		print("Freeze Language model")
		for name, param in self.model.language_model.named_parameters():
			param.requires_grad = False
		self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
		
	def train(self):
		loader = self.get_loader(split='train')
		loss_meter = LossMeter()
		print('Training')
		for epoch in range(5):
			for batch in (loader):
				results = self.train_step(batch)
				loss_meter.update(results['loss'].item())
			desc_str = f'Epoch {epoch} | Loss {loss_meter.val:.4f} |'
			print(desc_str)
			self.save(self.task + "_BEST")

	def train_step(self, batch):
		self.model.train()
		pixel_values = batch['pixel_values'].to(self.device) # bs, 36, 2048
		input_ids = batch['input_ids'].to(self.device) # bs, 20
		labels = batch['target_ids'].to(self.device)
		self.optimizer.zero_grad()
		output = self.model(input_ids=input_ids, pixel_values=pixel_values,labels=labels,)
		loss = output.loss
		loss.backward()
		self.optimizer.step()
		return {'loss': loss}

	def save(self, ckpt_name):
		savepath = os.path.join(self.args.output, f'{ckpt_name}.pth')
		state_dict_to_save={}
		state_dict_to_save["model"] = {
		    'query_tokens':self.model.query_tokens.data, 
		    'language_projection':self.model.language_projection.state_dict()}
		print(f'saving mode @ {savepath}')
		torch.save(state_dict_to_save, savepath)

	def load(self, ckpt_name):
		savepath = os.path.join(self.args.output, f'{ckpt_name}.pth')
		checkpoint = torch.load(savepath)
		print(f'Loading model from {savepath}')
		self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
		self.model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
		
			
	def get_loader(self, split='train'):
		dataset = QuestionDataset(task=self.task, split=split)
		loader = DataLoader(dataset, batch_size=32, 
			shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
		return loader

	def inference(self, data, task):
		self.load('object_BEST')
		self.model.eval()
		idx = 0
		for _d in data:
			entities = _d['entities']
			img_id = _d['image_id']
			image_dir = 'gvqa'
			image_path = f"{os.path.join(f'../datasets/{image_dir}', img_id)}.jpg"
			prompts, _ = self.generate_prompts(_d, task)
			image = Image.open(image_path).convert("RGB")
			inputs = self.processor([image] * len(prompts), text=prompts, truncation=True, padding=True, return_tensors="pt", max_length=60).to(self.device)
			# pixel_values = dataset[idx]['pixel_values'].to(self.device)
			# batch = {'pixel_values': inputs["pixel_values"], 'input_ids': inputs['input_ids']}
			generated_ids = self.model.generate(**inputs, max_new_tokens=10, num_beams=5, temperature=0.9, repetition_penalty=1.2)
			question = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
			import pdb;pdb.set_trace()
	def generate_prompts(self, datum, task):
		prompts, answers = [], []
		caption = datum['caption']
		entities = datum['entities']
		exclude_keys = {'Activity', 'Color', 'Material', 'Weather', 'Note'}
		# exclude_keys = {'Weather', 'Note', 'Category'}
		if task == "relation":
			entities['yes/no'] = ['yes', 'no']
		for key, value in entities.items():
			if key not in exclude_keys:
				item = random.choice(value)
				item = ' '.join(item.split()[:2])
				prompt = f"Caption:{caption} Answer:{item}. Question:"
				prompts.append(prompt)
				answers.append(item)
		return prompts, answers
if __name__ == "__main__":
	args = parse_args()
	args.output = 'snap/naiveblip_sgvqa_rev_cap'
	All_task = ["object", "attribute", "relation", "logical", "knowledge", "scenetext"]
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
	task = All_task[task_idx]
	trainer = QuestionTrainer(args, task)
	# trainer.train()
	data_path = os.path.join('../datasets/npy_cap_all', 'function', 'fcl_mmf_attribute_train.json')
	with open(data_path, 'r') as f:
		data = json.load(f)
	trainer.inference(data, 'object')

