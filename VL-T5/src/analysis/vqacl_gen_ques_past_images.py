import json
import os
import spacy
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2Config
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args
import sys
sys.path.insert(0, '../')
from Question_type import *
nlp = spacy.load("en_core_web_sm")
args = parse_args()

class GenQues:
	def __init__(self, savepath):
		self.savepath = savepath
		model_name = "Salesforce/blip2-opt-2.7b"
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.processor, self.model = self.initialize_model(model_name)

	def initialize_model(self, model_name):
		processor = AutoProcessor.from_pretrained(model_name)
		config = Blip2Config.from_pretrained(model_name)
		model = NaiveBLIP2.from_pretrained(model_name, config=config).to(self.device)
		return processor, model

	def _load_model(self, task):
		ckpt_path = os.path.join(self.savepath, f'{task}_LAST.pth')
		print(f'Loading checkpoint from @ {ckpt_path}')
		if os.path.exists(ckpt_path):
			checkpoint = torch.load(ckpt_path, map_location=self.device)
			self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
			self.model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])

	def _load_answers(self, task):
		path = f'../datasets/clove_stats/{task}_answer_stats.json'
		with open(path, 'r') as f:
			answer_dict = json.load(f)

		# Determine the number of top answers to keep based on the task type
		if task in ['object', 'attribute']:
			k = 150
		elif task == 'relation':
			k = 20
		elif task == 'logical':
			k = 2
		

		# Sort the dictionary by their frequencies (values) in descending order
		sorted_answers = sorted(answer_dict, key=answer_dict.get, reverse=True)

		# Store only the top k answers
		self.answer_list = sorted_answers[:k]


	def clean_question(self, output):
		# Define the keyword to identify the part of the string to remove
		keyword = "Question:"
		output = output.strip()
		# Split the output on the keyword
		parts = output.split(keyword)

		# Determine the part to use
		remaining_part = parts[-1] if len(parts) > 1 else output

		# Check if the remaining part starts with 'by' and handle accordingly
		remaining_part = remaining_part.strip().lower()
		if remaining_part.startswith('by') or remaining_part.startswith('for'):
			# Skip the first word
			return ' '.join(remaining_part.split(' ')[1:])
		else:
			# Return the trimmed text if it doesn't start with 'by'
			return remaining_part

	def inference_qa(self, image_path, datum, task, max_new_tokens=25):
		"""
		Generate question-answer pairs based on the provided image and method.
		"""
		try:
			image = Image.open(image_path).convert("RGB")
		except Exception as e:
			print(f"Error opening image: {e}")
			return []
		inputs = self.processor(image, truncation=True, padding=True, return_tensors="pt", max_length=32).to(self.device)
		batch = {'pixel_values': inputs["pixel_values"]}
		output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
		question, answer = output['questions'][0].split('?')
		return [(question.strip()+'?', answer.strip())]


	def generate_prompts(self, entities, task, caption=None):
		"""
		Generate prompts from entities with optional captions.
		"""
		prompts, answers = [], []
		for entity in set(entities):  # Remove duplicates here
			if entity in self.answer_list:
				answer = entity
				prompt = f"Caption:{caption} Answer:{answer}. Question:" if caption else f"Answer:{answer}. Question:"
				prompts.append(prompt)
				answers.append(answer)
		return prompts, answers


	def fallback_question_generation(self, inputs):
		print("Fallback")
		output = self.model.get_questions({'pixel_values': inputs["pixel_values"][0].unsqueeze(0)}, max_new_tokens=20)
		pred = output['questions'][0]
		temp, question = pred.split('Question:')
		answer = temp.split('Answer:')[-1]
		return [(question.strip(), answer.strip())]

def _load_data(root, task_idx):
	root = "../datasets/vqa/Partition_Q_V2/karpathy_train_"
	json_path = root + All_task[task_idx] + '.json'

	Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
	with open(json_path, 'r') as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	if task_idx != 9:
		each_memory = 20000
	else:
		each_memory = len(data_info_dicts)
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
	All_data = []
	for key in Examplar_set:
		All_data += Examplar_set[key]
	
	return All_data

if __name__ == "__main__":
	path = "../datasets/vqa/Partition_Q_V2/"
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9)) 
	task = All_task[task_idx]

	fname = f"karpathy_train_{task}.json"
	source = os.path.join(path, fname)
	dest_dir = f'../datasets/vqa/Partition_Q_V2_no_ents_past/'
	os.makedirs(dest_dir, exist_ok=True)
	dest = os.path.join(f'{dest_dir}', fname)
	savepath = f'snap/naiveblip_cl_no_ents/'  
	gen_ques = GenQues(savepath)
	incorrect = 0
	new_data = []
	# if task != 'q_causal':
	partition = int(20000/task_idx)
	# else:
	# 	partition = int(len(data)/task_idx)
	for i in range(task_idx):
		data = _load_data(path, i)
		qg_task = All_task[i]
		# start_idx = i * partition
		# end_idx = start_idx + partition
		# print(f"start idx: {start_idx}")
		# print(f"end idx: {end_idx}")
		print(f"{partition} sampled from {qg_task}")
		data_subset = random.sample(data, partition)
		gen_ques._load_model(qg_task)
		# gen_ques._load_answers(qg_task)
		print(f'Number of samples: {len(data_subset)}')
		for _d in tqdm(data_subset):
			img_name = f"{_d['img_id']}.jpg"
			split = "train" if 'train2014' in img_name else 'val'
			img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
			pairs = gen_ques.inference_qa(img_path, _d, qg_task)
			if pairs != None and len(pairs)>0 :
				questions, answers = zip(*pairs)
				if not "" in answers or not "" in answers:
					_d[f'Q_{qg_task}'] = questions
					_d[f'A_{qg_task}'] = answers
					new_data.append(_d)
			else:
				incorrect +=1
	print(f"Incorrect: {incorrect} in {len(data)} samples")
	with open(dest, 'w') as f:
		json.dump(new_data, f, indent=4)
