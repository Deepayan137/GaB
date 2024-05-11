import json
import os
import spacy
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2Config, AutoTokenizer
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args

# nlp = spacy.load("en_core_web_sm")
args = parse_args()

class ConfidenceEstimation():
	def __init__(self, ckpt_path):
		model_name = "Salesforce/blip2-opt-2.7b"
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.processor = AutoProcessor.from_pretrained(model_name)
		config = Blip2Config.from_pretrained(model_name)
		self.model = NaiveBLIP2.from_pretrained(model_name, config=config)
		self.model.to(self.device)
		if os.path.exists(ckpt_path):
			checkpoint = torch.load(ckpt_path, map_location=self.device)
			self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
			self.model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection_answers'])
			self.model.load_state_dict(checkpoint['model'])

	
	def get_predicted_answer(self, inputs):
		result = self.model.test_step(inputs, 'q_recognition')
		return result['pred_ans']

	def uncertainty_log_prob(self, inputs):
		with torch.no_grad():
			outputs = self.model.train_step(inputs, 0)
			logits = outputs['logits']
			decoded_answers = self.get_predicted_answer(inputs)
			max_probs = logits.softmax(dim=-1).max(dim=-1)[0]  # Max probability
			confidence = max_probs.mean().item()  # Mean max probability as confidence score
			return decoded_answers, confidence

	def uncertainty_entropy(self, inputs):
		with torch.no_grad():
			outputs = self.model.train_step(inputs, 0)
			logits = outputs['logits']
			probs = logits.softmax(dim=-1)
			decoded_answers = self.get_predicted_answer(inputs)
			entropy = -(probs * probs.log()).sum(dim=-1)  # Entropy of the probability distribution
			confidence = (1 / entropy.mean()).item()  # Inverse of mean entropy as confidence
			return decoded_answers, confidence

	def uncertainty_mc_dropout(self, inputs, num_samples):
		self.model.train()  # Enable dropout
		logits_list = []
		
		for _ in range(num_samples):
			with torch.no_grad():
				outputs = self.model.train_step(inputs, 0)
				logits_list.append(outputs.logits)

		mean_logits = torch.stack(logits_list).mean(0)
		decoded_answers = decoded_answers = self.get_predicted_answer(inputs)
		confidence = mean_logits.softmax(dim=-1).max(dim=-1)[0].mean().item()  # Mean max probability as confidence score
		self.model.eval()  # Disable dropout
		return decoded_answers, confidence

	def get_uncertainity(self, image_path, question, uncertainty=None):
		image = Image.open(image_path).convert("RGB")
		inputs = self.processor(image, question, return_tensors="pt", truncation=True).to(self.device)

		if uncertainty == 'mc_dropout':
			return self.uncertainty_mc_dropout(inputs, 10)
		elif uncertainty == 'entropy':
			return self.uncertainty_entropy(inputs)
		else:  # default to log_prob
			return self.uncertainty_log_prob(inputs)

if __name__ == "__main__":
	path = "../datasets/vqa/Partition_Q_V2/"
	fname = "karpathy_train_q_judge.json"
	source = os.path.join(path, fname)
	ckpt_path = 'snap/naiveblip_cl_syn_self/q_location_LAST.pth'
	con_est = ConfidenceEstimation(ckpt_path)
	
	with open(source, 'r') as f:
		data_subset = json.load(f)
	
	# for _d in tqdm(data_subset):
	img_id = 'COCO_train2014_000000404004'
	split = "train" if 'train2014' in img_id else 'val'
	img_path = os.path.join(f"../datasets/COCO/{split}2014", f"{img_id}.jpg")
	question = "where is the pizza?"
	# real_answer = gen_ques.get_pred_ans(img_path, question)
	# answer, entropy = gen_ques.get_uncertainity(img_path, question, uncertainty='entropy')  # Choose 'log_prob', 'entropy', or 'mc_dropout'
	answer, log_prob = con_est.get_uncertainity(img_path, f"Question: {question} Answer:")
	# _, mc = gen_ques.get_uncertainity(img_path, question, uncertainty='mc_dropout')
	print(f"Answer: {answer}, Confidence: {log_prob}")
	import pdb;pdb.set_trace()