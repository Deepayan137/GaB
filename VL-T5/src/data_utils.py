import os
import json
import random
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from src.vqa_data_blip import VQAEvaluator, get_loader_memory, VQADataset
sys.path.insert(0, '../')
from Question_type import *
from torchvision.transforms.functional import to_tensor

import re
import transformers
import logging
from tqdm import *

import warnings
warnings.filterwarnings('ignore')
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Your code follows here...

MODEL_DIR="./llama-2-7b-chat-hf"
All_task = ['q_recognition','q_location', 'q_judge', 'q_commonsense', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory', 'q_causal']

def get_memory_data(args, task_idx, each_memory, Examplar_set, model, processor):
	print("Welcome to the rehearsal memory module")
	if args.use_gen_data:
		print("We will use Synthetic QA pairs")
		if args.create_gen_data:
			task = All_task[task_idx]
			dest = '../datasets/vqa/Partition_Q_V2_subset_ST/'
			if not os.path.exists(f'{dest}/karpathy_train_{task}.json'):
				print(f"Synthetic QA pairs not found so creating  for task {task}")
				create_rehearsal_data(args, task_idx, model, processor, dest)
				print("Data creation completed, will load blip generated QA pairs")
		else:
			print("Loading Llama generated QA pairs")
			dest = '../datasets/vqa/Partition_Q_V2_llamaQA'
		each_memory = 5000
		Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
		data_info_path = (f'{dest}/karpathy_train_' + f'{All_task[task_idx]}.json')
	else:
		print("Loading real QA pairs from previos tasks")
		dest = '../datasets/vqa/Partition_Q_V2/'
		data_info_path = (f'{dest}/karpathy_train_' + f'{All_task[task_idx - 1]}.json')
	with open(data_info_path) as f:
		data_info_dicts = json.load(f)
	random.shuffle(data_info_dicts)
	if args.use_class_hierarchy:
		each_memory_for_cate = int(each_memory / len(Category_splits))
		for cate in Category_splits:
			num = 0
			if not args.use_gen_data:
				Examplar_set[cate].append([])
			for _d in data_info_dicts:
				img_id = _d['img_id']
				if img_id in ImgId_cate_map:
					if ImgId_cate_map[img_id] in Category_splits[cate]:
						if not args.use_gen_data:
							Examplar_set[cate][task_idx - 1].append(_d)
						else:
							Examplar_set[cate].append(_d)
						num += 1
						if num >= each_memory_for_cate:
							break
		print('Load from Partition_Q_v3......')
		if not args.use_gen_data:
			for cate in Category_splits:
				for i in range(task_idx):
					Examplar_set[cate][i] = Examplar_set[cate][i][: each_memory_for_cate]
			All_examplar = []
			for E_set in Examplar_set:
				for task_set in Examplar_set[E_set]:
					All_examplar += task_set
		else:
			All_examplar = []
			for E_set in Examplar_set:
				All_examplar += Examplar_set[E_set]
	else:
		All_examplar = data_info_dicts[:each_memory]
	print("# The size of the cate Memory:", len(All_examplar))			
	return All_examplar, Examplar_set

def post_process_answer(answer):
	answer = answer.strip().lower()  # Normalize the case to handle mixed cases
	corrections = {
		'ele': 'elephant',
		'gaffe': 'giraffe',
		'wich': 'sandwich',
		'sur': 'surfing',
		'fing': 'surfing',
		'nis': 'tennis',
		'ite': 'kite'
	}

	# Remove 'not' if it is at the end
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

class QAGen():
	def __init__(self, args, model, processor):
		self.args = args
		self.device = model.device
		self.savepath = args.output
		self.model = model
		self.processor = processor
		self.model.eval()

	def postprocess(self, sents):
		sents = sents.split('\n')
		questions, answers = [], []
		for sent in sents:
			if 'Q:' in sent:
				question, answer = sent.split('?')
				question = question.split(':')[-1].strip()
				answer = answer.split(':')[-1].strip()
				answer = post_process_answer(answer)
				if question.strip() != "" and answer != "":
					questions.append(question)
					answers.append(answer)
		return questions, answers

	def _load_model(self, task):
		ckpt = torch.load(os.path.join(self.savepath, f'{task}_LAST.pth'))
		print(f"question and answer gen projection head loaded for task {task} from {self.savepath}")
		self.model.language_projection_questions.load_state_dict(ckpt['model']['language_projection_questions'])
		# self.model.language_projection_answers.load_state_dict(ckpt['model']['language_projection_answers'])
	

	def generate(self, data, task, batch_size=32):
		entries = {}
		count = 0
		self._load_model(task)
		# data=data[:32]
		self.args.use_gen_data = False
		loader = get_loader_memory(self.args, All_task, data, batch_size=batch_size)
		for key in loader.keys():
			loader_cate = loader[key]
			for i, batch in enumerate(tqdm(loader_cate)):
				outputs = self.model.get_questions(batch)
				qids = batch['question_ids']
				generated_qa = outputs['questions']
				questions = [f"{item.split('?')[0]}?" for item in generated_qa]
				cap_answers = [item.split('?')[1] if len(item.split('?')) > 1 else "" for item in generated_qa]
				sents = [f"Question: {question} Answer:" for question in questions]
				input_ids = self.processor.tokenizer(text=sents, max_length=20, truncation=True, padding=True, return_tensors='pt')
				batch['input_ids'] = input_ids['input_ids']
				outputs = self.model.test_step(batch, task)
				answers = outputs['pred_ans']
				try:
					for i, qid in enumerate(qids):
						entries[qid] = (questions[i], answers[i], cap_answers[i])
					count += len(qids)
				except Exception as e:
					logging.info(f"Err processing batch {i+1}: {e}")
					continue
		print("Setting the old flag back")
		self.args.use_gen_data = True # restoring the gen data flag, this is only temporary, need to find an efficient soln
		return entries
	

	
	def _load_data(self, task_idx):
		each_memory = 10000
		root = "../datasets/vqa/Partition_Q_V2/karpathy_train_"
		json_path = root + All_task[task_idx] + '.json'
		Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
		with open(json_path, 'r') as f:
			data_info_dicts = json.load(f)
		random.shuffle(data_info_dicts)
		if self.args.use_class_hierarchy:
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
		return Examplar_set

def create_rehearsal_data(args, task_idx, model, processor, dest):
	os.makedirs(dest, exist_ok=True)
	qagen = QAGen(args, model, processor)
	task = All_task[task_idx]
	Examplar_set = qagen._load_data(task_idx)
	split = int(5000/task_idx)
	cat_split = int(split / 5)
	new_data = []
	incorrect_samples=0
	total_samples = 5000
	limit = 5000// task_idx
	total=0
	for i in range(task_idx):
		qg_task = All_task[i]
		print(f"Now task is {task} and question generation will be from {qg_task}")
		start_idx = i * cat_split
		end_idx = start_idx + cat_split
		limit_cat = limit // 5
		print(f"start idx: {start_idx}")
		print(f"end idx: {end_idx}")
		print(f"limit is: {limit}")
		All_examplar = []
		for key in Examplar_set.keys():
			All_examplar += Examplar_set[key][start_idx:end_idx]
		random.shuffle(All_examplar)
		entries = qagen.generate(All_examplar, qg_task, batch_size=32)
		# Serialize and save the data
		count = 0
		for examplar in All_examplar:
			qid = examplar['question_id']
			question, answer, caption_answer = (entry.strip() for entry in entries[qid])

			# Default answers if empty after stripping
			answer = answer or "not sure"
			caption_answer = caption_answer or "not sure"

			# Only process further if both answers are not "not sure" and are identical
			# if answer and caption_answer != "not sure" and caption_answer == answer:
			examplar[f"Q_{qg_task}"] = question
			examplar[f"A_self_{qg_task}"] = post_process_answer(answer)
			examplar[f"A_cap_{qg_task}"] = caption_answer
			new_data.append(examplar)
			count += 1
			total += 1
	
			if count >= limit:
				break
		
	with open(f'{dest}/karpathy_train_{task}.json', 'w') as json_file:
		json.dump(new_data, json_file, indent=4)
	print("Finished\n")
	# err_fr = (incorrect_samples/total_samples)*100
	print(f"Total number of samples:{total}")
	# print(f"num of correct_samples:{count}")
	# print(f"% of incorrect_samples:{err_fr}")
	# print(f"Total samples: {total_samples}")



if __name__ == "__main__":
	task_idx = 1
	backbone = "Salesforce/blip2-opt-2.7b"
	from src.param import parse_args
	args = parse_args()
	from transformers import Blip2Config
	config = Blip2Config.from_pretrained(backbone)
	from transformers import AutoProcessor
	from src.vqa_model_blip import NaiveBLIP2
	model = NaiveBLIP2.from_pretrained(backbone, config=config)
	processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
	savepath = 'snap/naiveblip_qa_qtoken/'
	args.backbone = backbone
	args.self_train = True
	dest = "/leonardo_scratch/fast/IscrC_CLRT-VLM/VQACL/datasets/vqa/Partition_Q_V2_st/"
	os.makedirs(dest, exist_ok=True)
	if task_idx > 0 :
		qagen = QAGen(args, model, processor, savepath)
		task = All_task[task_idx]
		Examplar_set = qagen._load_data(task_idx)
		split = int(5000/task_idx)
		cat_split = int(split / 5)
		new_data = []
		incorrect_samples=0
		total_samples = 5000
		for i in range(task_idx):
			qg_task = All_task[i]
			print(f"Now task is {task} and question generation will be from {qg_task}")
			start_idx = i * cat_split
			end_idx = start_idx + cat_split
			print(f"start idx: {start_idx}")
			print(f"end idx: {end_idx}")
			All_examplar = []
			for key in Examplar_set.keys():
				All_examplar += Examplar_set[key][start_idx:end_idx]
			entries = qagen.generate(All_examplar, qg_task, batch_size=32)
			# Serialize and save the data
			for _d in All_examplar:
				qid = _d['question_id']
				(question, answer) = entries[qid]
				if len(question) > 0 and len(answer) > 0:
					_d[f"Q_{qg_task}"] = question
					answer = post_process_answer(answer)
					_d[f"A_{qg_task}"] = answer
					new_data.append(_d)
				else:
					incorrect_samples +=1
		
		# with open(f'{dest}/karpathy_train_{task}.json', 'w') as json_file:
		# 	json.dump(new_data, json_file, indent=4)
		# logging.info("Finished\n")
		# err_fr = (incorrect_samples/total_samples)*100
		# logging.info(f"Total number of samples:{total_samples}")
		# logging.info(f"num of incorrect_samples:{incorrect_samples}")
		# logging.info(f"% of incorrect_samples:{err_fr}")
		# logging.info(f"Total samples: {total_samples}")




