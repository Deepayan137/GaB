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
	def __init__(self, savepath=None, model=None, processor=None):
		self.savepath = savepath
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if model is not None and processor is not None:
			self.model, self.processor = model, processor
		else:
			model_name = "Salesforce/blip2-opt-2.7b"
			
			self.processor, self.model = self.initialize_model(model_name)
		
	def initialize_model(self, model_name):
		processor = AutoProcessor.from_pretrained(model_name)
		config = Blip2Config.from_pretrained(model_name)
		model = NaiveBLIP2.from_pretrained(model_name, config=config).to(self.device)
		return processor, model

	def _load_model(self, task):
		ckpt_path = os.path.join(self.savepath, f'{task}_LAST.pth')
		if os.path.exists(ckpt_path):
			print(f'Loading checkpoint from @ {ckpt_path}')
			checkpoint = torch.load(ckpt_path, map_location=self.device)
			self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
			self.model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])
			self.model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection_answers'])
	
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

	def _load_qtype_stats(self, task, desired_counts):
		elements, weights = zip(*desired_counts.items())
		weights = [w / sum(weights) for w in weights]  # Normalize weights to sum to 1
		sampled_element = random.choices(elements, weights=weights, k=1)[0]
		return sampled_element
		
	def inference_qa(self, image_path, datum, task, desired_counts, method, max_new_tokens=25, self_answer=False):
		"""
		Generate question-answer pairs based on the provided image and method.
		"""
		try:
			image = Image.open(image_path).convert("RGB")
		except Exception as e:
			print(f"Error opening image: {e}")
			return []
		
		if method == 'qtype':
			qtype = self._load_qtype_stats(task, desired_counts['balanced'])
			prompt = f"Question Type:{qtype} Question:"
			inputs = self.processor(image, text=prompt,truncation=True, padding=True, return_tensors="pt", max_length=32).to(self.device)
			batch = {'pixel_values': inputs["pixel_values"], 'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
			output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
			try:
				question, answer = output['questions'][0].split('Answer:')
				return [(question.strip(), answer.strip())]
			except :
				print("ooops")
				return None
		elif method == "lamol":
			prompt = f"{task}:"
			inputs = self.processor(image, text=prompt,truncation=True, padding=True, return_tensors="pt", max_length=32).to(self.device)
			batch = {'pixel_values': inputs["pixel_values"], 'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
			output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
			caption = output['questions'][0]
			if 'answer:' in caption:
				try:
					question, answer = output['questions'][0].split('answer:')
					return [(f'{task}:'+question.strip()+'?', answer.strip())]
				except:
					return None
			try:
				question, answer = output['questions'][0].split('?')
				return [(f'{task}:'+question.strip(), answer.strip())]
			except :
				print("ooops")
				return None
		elif method == "vqg":
			prompt = f"{answer}:"
			inputs = self.processor(image, text=prompt,truncation=True, padding=True, return_tensors="pt", max_length=32).to(self.device)
			batch = {'pixel_values': inputs["pixel_values"], 'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
			output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
			try:
				question, answer = output['questions'][0].split('?')
				return [(question.strip(), answer.strip())]
			except :
				print("ooops")
				return None
		else:
			inputs = self.processor(image, truncation=True, padding=True, return_tensors="pt", max_length=32).to(self.device)
			batch = {'pixel_values': inputs["pixel_values"]}
			output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
			question, answer = output['questions'][0].split('?')
			
			if self_answer:
				sents = f"Question: {question} Answer:"
				input_ids = self.processor.tokenizer(text=sents, max_length=40, truncation=True, padding=True, return_tensors='pt')
				batch['input_ids'] = input_ids['input_ids']
				outputs = self.model.test_step(batch, task)
				answer = post_process_answer(outputs['pred_ans'][0])

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

def _load_data(root, task_idx, mem_size):
	root = "../datasets/vqa/Partition_Q_V2/karpathy_train_"
	json_path = root + All_task[task_idx] + '.json'

	Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
	with open(json_path, 'r') as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	if task_idx != 9:
		each_memory = mem_size
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

def post_process_answer(answer):
	answer = answer.strip().lower()  # Normalize the case to handle mixed cases
	corrections = {
		'ele': 'elephant',
		'gaffe': 'giraffe',
		'wich': 'sandwich',
		'sur': 'surfing',
		'fing': 'surfing',
		'nis': 'tennis',
		'ite': 'kite',
	}

	# Remove 'not' if it is at the end
	if answer.startswith('in'):
		answer = answer.replace('in', 'in ')
	if answer.startswith('on'):
		answer = answer.replace('on', 'on ')
	if answer.endswith('not'):
		answer = answer[:-3].strip()

	# Apply corrections based on exact matches
	if answer in corrections:
		return corrections[answer]

	# Handle cases starting with specific substrings
	if answer.startswith('bee'):
		return 'frisbee'
	if answer.startswith('ate') and answer.endswith('boarding'):
		return 'skateboarding'  # Corrected spelling

	return answer

if __name__ == "__main__":
	path = "../datasets/vqa/Partition_Q_V2/"
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 4)) 
	task = All_task[task_idx]
	method = 'lamol'
	savepath = f'snap/naiveblip_cl_{method}/' 
	gen_ques = GenQues(savepath)
	mem_size = 20000
	if method == 'lamol':
		mem_size = 5000
		gen_ques._load_model(task)
	self_answer = False
	if self_answer:
		print("Answers will be self generated")
	fname = f"karpathy_train_{task}.json"
	source = os.path.join(path, fname)
	dest_dir = f'../datasets/vqa/Partition_Q_V2_{method}/'
	print(f"Files will be saved @ {dest_dir}")
	os.makedirs(dest_dir, exist_ok=True)
	dest = os.path.join(f'{dest_dir}', fname)
	data = _load_data(path, task_idx, mem_size)
	with open(f'metrics/{task}_question_dist.json', 'r') as f:
		desired_counts = json.load(f)
	
	incorrect = 0
	new_data = []
	if task != 'q_causal':
		partition = int(mem_size/task_idx)
	else:
		partition = int(len(data)/task_idx)
	for i in range(task_idx):
		qg_task = All_task[i]
		start_idx = i * partition
		end_idx = start_idx + partition
		print(f"Now task is {task} and question generation will be from {qg_task}")
		print(f"start idx: {start_idx}")
		print(f"end idx: {end_idx}")
		data_subset = data[start_idx:end_idx]
		if method != 'lamol':
			gen_ques._load_model(qg_task)
		# gen_ques._load_answers(qg_task)
		print(f'Number of samples: {len(data_subset)}')
		for _d in tqdm(data_subset):
			img_name = f"{_d['img_id']}.jpg"
			split = "train" if 'train2014' in img_name else 'val'
			img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
			pairs = gen_ques.inference_qa(img_path, _d, qg_task, desired_counts[qg_task], method, self_answer=False)
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
