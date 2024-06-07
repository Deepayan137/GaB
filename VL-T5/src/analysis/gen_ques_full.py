import json
import os
import spacy
import copy
import torch
import numpy as np
import random
from PIL import Image
from tqdm import *
from transformers import AutoProcessor, Blip2Config
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args
from src.analysis.question_distribution import classify_questions, _load_classifier_ckpt
from src.analysis.question_distribution_clustering import cluster_questions
from src.analysis.question_classifier import QuestionTypeClassifier
import sys
sys.path.insert(0, '../')
from Question_type import Sg_task, qtype_dict
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
		else:
			print("Path does not exist")
	
	def inference_qa(self, image_path, datum, task, max_new_tokens=25, method='no_ents'):
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


def main():
	# Configuration and setup
	classify_strategy = "cluster"
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 4))
	method = 'no_ents'
	task = Sg_task['function']['oarlks'][task_idx]
	fname = f"fcl_mmf_{task}_train.npy"
	npy_path = os.path.join('../datasets/npy/function', fname)
	data = np.load(npy_path, allow_pickle=True)
	data = data[:50]  # Taking a slice for testing
	dest_dir = f'../datasets/npy_{method}/function'
	os.makedirs(dest_dir, exist_ok=True)
	dest_file = os.path.join(dest_dir, fname.replace('.npy', '_full.json'))
	save_path = f'snap/naiveblip_sgvqa_{method}/'
	
	gen_ques = GenQues(save_path)
	incorrect_count = 0
	processed_data = {}
	final_data = {}
	input_dim = 768
	hidden_dim = 256
	# Data processing
	sub_task_questions = {}
	for idx in trange(task_idx):
		qg_task = Sg_task['function']['oarlks'][idx]
		processed_data[qg_task] = []
		final_data[qg_task] = []
		output_dim = len(qtype_dict[qg_task])
		classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
		classifier = _load_classifier_ckpt(classifier, qg_task)
		gen_ques._load_model(qg_task)
		print(f'Processing {len(data)} samples for task: {qg_task}')
		sub_task_questions[qg_task] = []
		for entry in tqdm(data):
			
			# Create a deep copy of entry to ensure unique modification per task
			task_entry = copy.deepcopy(entry)
			img_name = f"{task_entry['image_id']}.jpg"
			img_path = os.path.join("../datasets/gvqa/", img_name)
			pairs = gen_ques.inference_qa(img_path, task_entry, qg_task, method=method)
			
			if pairs:
				questions, answers = zip(*pairs)
				task_entry[f'Q_{qg_task}'] = questions
				task_entry[f'A_{qg_task}'] = answers
				processed_data[qg_task].append(task_entry)
				sub_task_questions[qg_task].append(questions[0])
			else:
				incorrect_count += 1
		if classify_strategy == "classifier":
			print("Classifying questions")
			predictions = classify_questions(classifier, sub_task_questions, qg_task)
		elif classify_strategy == "cluster":
			print("Clustering questions")
			predictions = cluster_questions(sub_task_questions, qg_task)
		for index, datum_dict in enumerate(processed_data[qg_task]):
			datum_dict[f'{classify_strategy}_prediction'] = str(predictions[index])
			final_data[qg_task].append(datum_dict)
	# Summary and saving output
	total_samples = len(data) * task_idx
	print(f"Incorrect: {incorrect_count} in {total_samples} samples")
	with open(dest_file, 'w') as file:
		json.dump(processed_data, file, indent=4)

if __name__ == "__main__":
	main()
		