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
from Question_type import Sg_task
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
		if os.path.exists(ckpt_path):
			checkpoint = torch.load(ckpt_path, map_location=self.device)
			self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
			self.model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])

	def inference_qa(self, image_path, datum, task, max_new_tokens=20):
		self._load_model(task)
		# if task != 'object' or not entities:
		# 	return []

		prompts, answers = self.generate_prompts(datum, task)
		if not prompts:
			return []

		image = Image.open(image_path).convert("RGB")
		inputs = self.processor([image] * len(prompts), text=prompts, truncation=True, padding=True, return_tensors="pt", max_length=20).to(self.device)
		batch = {'pixel_values': inputs["pixel_values"], 'input_ids': inputs['input_ids']}
		output = self.model.get_questions(batch, max_new_tokens=max_new_tokens)
		pred_questions = output['questions']
		import pdb;pdb.set_trace()
		pairs = []
		for question, answer in zip(pred_questions, answers):
			if question and question != '.':
				if question.startswith(' place') or question.startswith(' animal') or question.startswith(' vehicle'):
					question = "which" + ' ' +question.strip()
				answer = answer.strip()
				pairs.append((question, answer))
		# pairs = [(q.strip(), a.strip()) for q, a in zip(pred_questions, answers) if q]
		if len(pairs) == 0:
			pairs = self.fallback_question_generation(inputs)
		return pairs

	def generate_prompts(self, datum, task):
		prompts, answers = [], []
		# exclude_keys = {'Activity', 'Color', 'Material', 'Weather', 'Note'}
		caption = datum['caption']
		entities = datum['entities']
		exclude_keys = {'Weather', 'Note', 'Category'}
		if task == "relation":
			entities['yes no'] = ['yes', 'no']
		for key, value in entities.items():
			if key not in exclude_keys:
				item = random.choice(value)
				item = ' '.join(item.split()[:2])
				prompt = f"Caption:{caption} Answer:{item}. Question:"
				prompts.append(prompt)
				answers.append(item)
		return prompts, answers

	def fallback_question_generation(self, inputs):
		print("Fallback")
		output = self.model.get_questions({'pixel_values': inputs["pixel_values"][0].unsqueeze(0)}, max_new_tokens=20)
		pred = output['questions'][0]
		temp, question = pred.split('Question:')
		answer = temp.split('Answer:')[-1]
		return [(question.strip(), answer.strip())]

if __name__ == "__main__":
	path = "../datasets/npy_cap_all/function"
	# task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) 
	task_idx = 1
	task = Sg_task['function']['oarlks'][task_idx]
	split = int(50)
	fname = f"fcl_mmf_{task}_train.json"
	source = os.path.join(path, fname)
	with open(source, 'r') as f:
		data = json.load(f)
	data=data[:50]
	savepath = 'snap/naiveblip_sgvqa_rev_cap_orig'	
	gen_ques = GenQues(savepath)
	incorrect = 0
	new_data = []
	for i in range(task_idx):
		qg_task = Sg_task['function']['oarlks'][i]
		start_idx = i * split
		end_idx = start_idx + split
		print(f"Now task is {task} and question generation will be from {qg_task}")
		print(f"start idx: {start_idx}")
		print(f"end idx: {end_idx}")
		data_subset = data[start_idx:end_idx]
		print(f'Number of samples: {len(data_subset)}')
		for _d in (data_subset):
			img_name = f"{_d['image_id']}.jpg"
			img_path = os.path.join("../datasets/gvqa/", img_name)
			pairs = gen_ques.inference_qa(img_path, _d, qg_task)
			if len(pairs)>0:
				questions, answers = zip(*pairs)
				_d[f'Q_ents_{qg_task}'] = questions
				_d[f'A_ents_{qg_task}'] = answers
				new_data.append(_d)
			else:
				incorrect +=1
	print(f"Incorrect: {incorrect} in {len(data_subset)} samples")
	with open(source, 'w') as f:
		json.dump(new_data, f, indent=4)
		