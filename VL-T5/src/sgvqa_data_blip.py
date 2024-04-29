import os
import sys
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from PIL import Image
import torch
import numpy as np
from copy import deepcopy
import re
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SGVQA(Dataset):
	def __init__(self, task='object', split='train', scenario='scene', verbose=True, args=None):
		super().__init__()
		filename = f'fcl_mmf_{task}_{split}.npy'
		data_path = os.path.join('../datasets/npy', scenario, filename)
		print(data_path)
		self.data = np.load(data_path, allow_pickle=True)
		# self.data = self.data[:50]

		self.args=args
		if 'blip' in self.args.backbone:
			self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		self.processor.tokenizer.padding_side = 'right'
		self.processor.tokenizer.truncation_side = 'right'
		if verbose:
			print("# all sentences:", len(self.data), 'with Examplers')
		self.split = split

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		out_dict = {}
		out_dict['args'] = self.args

		datum = self.data[idx]
		img_id = datum['image_id']
		out_dict['img_id'] = img_id
		if datum['image_source'] == 'vg':
			image_dir = 'gvqa'
		elif datum['image_source'] == 'textvqa':
			image_dir = f"textvqa_train"
		f = f"{os.path.join(f'../datasets/{image_dir}', img_id)}.jpg"
		if os.path.exists(f):
			image = Image.open(f).convert("RGB")
		else:
			raise "image path does not exists"
		if 'question' in datum:
			sent = datum['question']
		max_length = 20
		sent = f"Question: {sent.lower()} Answer:"
		inputs = self.processor(image, text=sent, max_length=max_length, 
			truncation=True, return_tensors="pt")
		out_dict['pixel_values'] = inputs['pixel_values']

		question_id = datum['question_id']
		out_dict['question_id'] = question_id

		out_dict['sent'] = sent
		out_dict['input_ids'] = inputs["input_ids"]
		out_dict['input_length'] = len(inputs["input_ids"][0])
		if 'answer' in datum:
			if isinstance(datum['answer'], list):
				answer = datum['answer'][0]
			else:
				answer = datum['answer']
			out_dict['answer'] = answer
		if 'answers' in datum:    
			out_dict['all_answers'] = datum['answers']
		target_ids = self.processor.tokenizer.encode(answer, max_length=10, truncation=True)
		out_dict['target_ids'] = torch.LongTensor(target_ids)
		out_dict['target_length'] = len(target_ids)
		return out_dict

	def collate_fn(self, batch):
		batch_entry = {}

		args = batch[0]['args']

		B = len(batch)
		S_W_L = max(entry['input_length'] for entry in batch)
		input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		if args.use_vision:
			vis_feats = torch.zeros(B, 3, 224, 224, dtype=torch.float)

		if 'target_ids' in batch[0]:
			T_W_L = max(entry['target_length'] for entry in batch)
			target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		
		sentences = []
		question_ids = []
		answers = []
		all_answers = []
		img_ids = []
		img_paths = []
		labels = []
		img_ids = []
		for i, entry in enumerate(batch):
			input_ids[i, :entry['input_length']] = entry['input_ids'][0]
			if 'target_ids' in entry:
				target_ids[i, :entry['target_length']] = entry['target_ids']
			if args.use_vision:
				vis_feats[i] += entry['pixel_values'][0]
			sentences.append(entry['sent'])
			question_ids.append(entry['question_id'])
			if 'answer' in entry:
				answers.append(entry['answer'])
			if 'all_answers' in entry:
				all_answers.append(entry['all_answers'])

			if 'img_id' in entry:
				img_ids.append(entry['img_id'])

		batch_entry['input_ids'] = input_ids
		if 'target_ids' in batch[0]:
			batch_entry['target_ids'] = target_ids

		if args.use_vision:
			batch_entry['pixel_values'] = vis_feats
		batch_entry['sent'] = sentences
		batch_entry['question_ids'] = question_ids
		batch_entry['answers'] = answers
		batch_entry['all_answers'] = all_answers
		batch_entry['img_id'] = img_ids
		batch_entry['args'] = args
		batch_entry['task'] = 'sgvqa'
		return batch_entry



class SGVQA_memory(Dataset):
	def __init__(self, Examplar_set, split='train', verbose=True, args=None):
		super().__init__()
		self.data = Examplar_set
		if verbose:
			print("# all sentences:", len(self.data), 'with Examplers')
		self.args=args
		if 'blip' in self.args.backbone:
			self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		self.processor.tokenizer.padding_side = 'right'
		self.processor.tokenizer.truncation_side = 'right'
		if verbose:
			print("# all sentences:", len(self.data), 'with Examplers')
		self.split = split

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		out_dict = {}
		out_dict['args'] = self.args

		datum = self.data[idx]
		img_id = datum['image_id']

		out_dict['img_id'] = img_id
		if datum['image_source'] == 'vg':
			image_dir = 'gvqa'
		elif datum['image_source'] == 'textvqa':
			image_dir = f"textvqa_{self.split}"
		f = f"{os.path.join(f'../datasets/{image_dir}', img_id)}.jpg"
		if os.path.exists(f):
			image = Image.open(f).convert("RGB")
		else:
			raise "image path does not exists"
		if 'question' in datum:
			sent = datum['question']
		max_length = 20
		sent = f"Question: {sent.lower()}? Answer:"
		inputs = self.processor(image, text=sent, max_length=max_length, 
			truncation=True, return_tensors="pt")
		out_dict['pixel_values'] = inputs['pixel_values']

		question_id = datum['question_id']
		out_dict['question_id'] = question_id

		out_dict['sent'] = sent
		out_dict['input_ids'] = inputs["input_ids"]
		out_dict['input_length'] = len(inputs["input_ids"][0])
		if 'answer' in datum:
			out_dict['answer'] = datum['answer']
		if 'answers' in datum:    
			out_dict['all_answers'] = datum['answers']
		target_ids = self.processor.tokenizer.encode(datum['answer'], 
			max_length=10, truncation=True)
		out_dict['target_ids'] = torch.LongTensor(target_ids)
		out_dict['target_length'] = len(target_ids)
		return out_dict

	def collate_fn(self, batch):
		batch_entry = {}

		args = batch[0]['args']

		B = len(batch)
		S_W_L = max(entry['input_length'] for entry in batch)
		input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		if args.use_vision:
			vis_feats = torch.zeros(B, 3, 224, 224, dtype=torch.float)
		if 'target_ids' in batch[0]:
			T_W_L = max(entry['target_length'] for entry in batch)
			target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.processor.tokenizer.pad_token_id
		
		sentences = []
		question_ids = []
		answers = []
		all_answers = []
		img_ids = []
		img_paths = []
		labels = []
		img_ids = []
		for i, entry in enumerate(batch):
			input_ids[i, :entry['input_length']] = entry['input_ids'][0]
			if 'target_ids' in entry:
				target_ids[i, :entry['target_length']] = entry['target_ids']
			if args.use_vision:
				vis_feats[i] += entry['pixel_values'][0]
			sentences.append(entry['sent'])
			question_ids.append(entry['question_id'])
			if 'answer' in entry:
				answers.append(entry['answer'])
			if 'all_answers' in entry:
				all_answers.append(entry['all_answers'])

			if 'img_id' in entry:
				img_ids.append(entry['img_id'])

		batch_entry['input_ids'] = input_ids
		if 'target_ids' in batch[0]:
			batch_entry['target_ids'] = target_ids

		if args.use_vision:
			batch_entry['pixel_values'] = vis_feats

		batch_entry['sent'] = sentences
		batch_entry['question_ids'] = question_ids
		batch_entry['answers'] = answers
		batch_entry['all_answers'] = all_answers
		batch_entry['img_id'] = img_ids
		batch_entry['args'] = args
		batch_entry['task'] = 'sgvqa'
		return batch_entry

def get_loader_memory(args, Examplar_set, split='train', scenario='scene',
			   batch_size=32, workers=4):

	verbose = True
	def cat_loader():
		dataset = SGVQA_memory(
			Examplar_set,
			split=split,
			verbose=verbose,
			args=args)

		
		if split == 'train':
			shuffle = len(dataset) > 0
			loader = DataLoader(dataset, 
				batch_size=batch_size, shuffle=shuffle,
				num_workers=workers, pin_memory=True,
				collate_fn=dataset.collate_fn)
		else:
			loader = DataLoader(
				dataset,
				batch_size=batch_size,
				num_workers=workers, pin_memory=True,
				shuffle=False,
				collate_fn=dataset.collate_fn)

		if verbose:
			loader.evaluator = SGVQAEvaluator()

		loader.task = 'sgvqa'
		return loader
	cate_loader = cat_loader()
	return cate_loader

def get_loader(args, split='train', scenario='scene',
			   batch_size=32, workers=4, task='object'):
	verbose=True
	def cat_loader():
		dataset = SGVQA(
			split=split,
			verbose=verbose,
			args=args,
			scenario=scenario,
			task=task)
		dataset_num = len(dataset)
		if split == 'train':
			loader = DataLoader(
				dataset, 
				batch_size=batch_size, 
				num_workers=workers, pin_memory=True,
				shuffle=True,
				collate_fn=dataset.collate_fn)
		else:
			loader = DataLoader(
				dataset,
				batch_size=batch_size,
				num_workers=workers, pin_memory=True,
				shuffle=False,
				collate_fn=dataset.collate_fn)
		loader.evaluator = SGVQAEvaluator()
		loader.task = 'sgvqa'
		return loader, dataset_num
	total_num = 0
	cate_loader, total_num = cat_loader()
	return cate_loader, total_num

def get_loader_test(args, split='train', scenario='scene',
			   batch_size=32, workers=4, task='what'):
	verbose=True
	dataset = SGVQA(
		split=split,
		verbose=verbose,
		args=args,
		scenario=scenario,
		task=task) # all categories

	
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=workers, pin_memory=True,
		shuffle=False,
		collate_fn=dataset.collate_fn)

  
	loader.evaluator = SGVQAEvaluator()

	loader.task = 'sgvqa'
	return loader

class SGVQAEvaluator:
	def __init__(self):

		"""https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

		self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

		self.articles     = ['a',
							 'an',
							 'the'
							]

		self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.commaStrip   = re.compile("(\d)(\,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

		self.n = 2

	def dump_result(self, quesid2ans: dict, path):
		"""
		Dump results to a json file, which could be submitted to the VQA online evaluation.
		VQA json file submission requirement:
			results = [result]
			result = {
				"question_id": int,
				"answer": str
			}
		:param quesid2ans: dict of quesid --> ans
		:param path: The desired path of saved file.
		"""
		with open(path, 'w') as f:
			result = []
			for ques_id, ans in quesid2ans.items():
				result.append({
					'question_id': ques_id,
					'answer': ans
				})
			json.dump(result, f, indent=4, sort_keys=True)

	def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
		self.accuracy     = {}
		self.evalQA       = {}

		accQA = []
		accQuesType = {}
		accAnsType = {}

		for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):
			resAns, gts = resAns
			resAns      = resAns.replace('\n', ' ')
			resAns      = resAns.replace('\t', ' ')
			resAns      = resAns.strip()
			resAns      = self.processPunctuation(resAns)
			resAns      = self.processDigitArticle(resAns)

			gtAcc  = []
			gtAnswers = gts
			ansDic={}
			gtAnswers = [self.processPunctuation(item) for item in gtAnswers]
			gtAnswers = list(enumerate(gtAnswers))

			for gtAnsDatum in gtAnswers:
				otherGTAns = [item for item in gtAnswers if item != gtAnsDatum]
				matchingAns = [item for item in otherGTAns if item[1] == resAns]
				acc = min(1, float(len(matchingAns))/3)
				gtAcc.append(acc)
			avgGTAcc = float(sum(gtAcc))/len(gtAcc)
			accQA.append(avgGTAcc)
			self.setEvalQA(quesId, avgGTAcc)
		if len(accQA) == 0:
			return {
				'overall': 0,
				'perQuestionType': {},
				'perAnswerType': {}
			}
		else:
			self.setAccuracy(accQA)
		return self.accuracy

	def normalize_answer(self, resAns):
		resAns      = resAns.replace('\n', ' ')
		resAns      = resAns.replace('\t', ' ')
		resAns      = resAns.strip()
		resAns      = self.processPunctuation(resAns)
		resAns      = self.processDigitArticle(resAns)
		resAns = resAns.replace(',', '')
		return resAns

	def processPunctuation(self, inText):
		outText = inText
		for p in self.punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')
		outText = self.periodStrip.sub("",
										outText,
										re.UNICODE)
		return outText

	def processDigitArticle(self, inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = self.manualMap.setdefault(word, word)
			if word not in self.articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in self.contractions:
				outText[wordId] = self.contractions[word]
		outText = ' '.join(outText)
		return outText

	def setEvalQA(self, quesId, acc):
		self.evalQA[quesId] = round(100*acc, self.n)

	def setAccuracy(self, accQA):
		self.accuracy['overall']  = round(100*float(sum(accQA))/len(accQA), self.n)


if __name__ == "__main__":
	from src.param import parse_args
	
	args = parse_args()
	args.backbone = 'blip'
	split = f'val'
	scenario='scene'
	task='a#ShopAndDining'
	data_info_path = ('../datasets/npy/scene/fcl_mmf_' + f'a#ShopAndDining_train.npy')
	data_info_dicts = np.load(data_info_path, allow_pickle=True)

	random.shuffle(data_info_dicts)  # shuffle
	All_examplar = data_info_dicts[:5000]
	loader = get_loader_test(args, scenario=scenario, task=task, workers=0)
	quesid2ans = {}
	gtAnswers = {}
	for batch in loader:
		
		answers = batch['answers']
		qids = batch['question_ids']
		all_answers = batch['all_answers']
		pairs = list(zip(qids, answers, all_answers))
		for qid, answer, all_ in pairs:
			quesid2ans[qid] = answer
			gtAnswers[qid] = {}
			gtAnswers[qid]['answers'] = all_ 
		acc = loader.evaluator.evaluate_raw(quesid2ans, gtAnswers)
		print(acc)