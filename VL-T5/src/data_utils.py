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

import os
import re
import json
import torch
import transformers
import logging
from tqdm import *
from transformers import LlamaForCausalLM, LlamaTokenizer
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR="./llama-2-7b-chat-hf"
All_task = ['q_recognition','q_location', 'q_judge', 'q_commonsense', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory', 'q_causal']

def get_memory_data(args, task_idx, each_memory, Examplar_set):
	if args.use_gen_data:
		each_memory = 5000
		Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
		data_info_path = ('../datasets/vqa/Partition_Q_V2_subset_new/karpathy_train_' + f'{All_task[task_idx]}.json')
	else:
		data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{All_task[task_idx - 1]}.json')
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

class QAGen():
	def __init__(self, args, model, processor, savepath):
		self.args = args
		self.device = 'cuda'
		self.savepath = savepath
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
				if question != "" and answer != "":
					questions.append(question)
					answers.append(answer)
		return questions, answers

	def _load_model(self, task):
		ckpt = torch.load(os.path.join(self.savepath, f'{task}_LAST.pth'))
		print(f"question gen projection head loaded for task {task} from {self.savepath}")
		self.model.language_projection_questions.load_state_dict(ckpt['model']['language_projection_questions'])
		self.model.language_projection_answers.load_state_dict(ckpt['model']['language_projection_answers'])
	

	def generate(self, data, task, batch_size=32):
		entries = {}
		count = 0
		self._load_model(task)
		data=data[:10]
		loader = get_loader_memory(self.args, All_task, data, batch_size=2)
		for key in loader.keys():
			loader_cate = loader[key]
			for i, batch in enumerate(loader_cate):
				outputs = self.model.get_questions(batch)
				qids = batch['question_ids']
				generated_qa = outputs['questions']
				questions = [f"{item.split('?')[0]}?" for item in generated_qa]
				# answers = [f"{item.split('?')[1]}" for item in generated_qa]
				if self.args.self_train:
					sents = [f"Question: {question} Answer:" for question in questions]
					input_ids = self.processor.tokenizer(text=sents, max_length=20, truncation=True, padding=True, return_tensors='pt')
					batch['input_ids'] = input_ids['input_ids']
					outputs = self.model.test_step(batch, task)
					answers = outputs['pred_ans']
				try:
					for i, qid in enumerate(qids):
						entries[qid] = (questions[i], answers[i])
					count += len(qids)
				except Exception as e:
					logging.info(f"Err processing batch {i+1}: {e}")
					continue
				# logging.info(f"{i+1} out of {total_batches} completed")
		return entries
	

	
	def _load_data(self, task_idx):
		each_memory = 5000
		root = "/leonardo_scratch/fast/IscrC_CLRT-VLM/VQACL/datasets/vqa/Partition_Q_V2/karpathy_train_"
		json_path = root + All_task[task_idx] + '.json'
		Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
		with open(json_path, 'r') as f:
			data_info_dicts = json.load(f)
		random.shuffle(data_info_dicts)
		if args.use_class_hierarchy:
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
			entries = qagen.generate(All_examplar, qg_task, batch_size=2)
			# Serialize and save the data
			for _d in All_examplar:
				qid = _d['question_id']
				(question, answer) = entries[qid]
				if len(question) > 0 and len(answer) > 0:
					_d[f"Q_{qg_task}"] = question
					_d[f"A_{qg_task}"] = answer
					new_data.append(_d)
				else:
					incorrect_samples +=1
		
		with open(f'{dest}/karpathy_train_{task}.json', 'w') as json_file:
			json.dump(new_data, json_file, indent=4)
		logging.info("Finished\n")
		err_fr = (incorrect_samples/total_samples)*100
		logging.info(f"Total number of samples:{total_samples}")
		logging.info(f"num of incorrect_samples:{incorrect_samples}")
		logging.info(f"% of incorrect_samples:{err_fr}")
		logging.info(f"Total samples: {total_samples}")




