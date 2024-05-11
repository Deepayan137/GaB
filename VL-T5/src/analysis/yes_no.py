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
nlp = spacy.load('en_core_web_sm')




class YesNo():
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
			self.model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection'])
			# self.model.load_state_dict(checkpoint['model'])
	
	def get_yes_no(self, image_path, question):
		image = Image.open(image_path).convert("RGB")
		inputs = self.processor(image, question, return_tensors="pt", truncation=True).to(self.device)
		pixel_values = inputs['pixel_values']
		query_outputs, vision_outputs = self.model.get_features(pixel_values)
		input_ids = inputs['input_ids']
		max_new_tokens = 20
		output = self.model.generate(
            query_outputs=query_outputs, 
            vision_outputs=vision_outputs, 
            input_ids=input_ids, 
            max_new_tokens=max_new_tokens, 
            repetition_penalty=1.2,
            mode='answers')
		pred_ans = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)
		return pred_ans
	
	def transform_color_question(self, question, answer):
	    doc = nlp(question)
	    # Identify the main subject following the color query
	    subject_parts = [token.text for token in doc if token.dep_ not in ['det', 'aux', 'attr']]
	    subject = ' '.join(subject_parts[3:])  # Skip the initial words of the question

	    if 'what color is the' in question.lower():
	        # Type 1 questions
	        return f"Is the color of the {subject} {answer}?"
	    elif 'what color is' in question.lower():
	        # Type 2 questions
	        return f"Is {subject} {answer}?"
	    elif 'what color are the' in question.lower():
	        # Type 3 questions
	        return f"Are the {subject} {answer}?"
	    elif 'what is the color of the' in question.lower():
	        # Type 4 questions
	        return f"Is the color of the {subject} {answer}?"
	    elif 'what color' in question.lower():
	        # Type 5 questions
	        return f"Is the {subject} {answer}?"
	    else:
	        # Fallback for unexpected formats
	        return f"Does the question '{question}' correspond to {answer}?"

	def transform_counting_question(self, question, answer):
	    doc = nlp(question)
	    start_phrase = ' '.join([token.text for token in doc[:4]])  # Capture enough of the start to classify the type

	    if start_phrase.lower().startswith('how many'):
	        subject = ' '.join([token.text for token in doc[2:] if token.dep_ != 'aux'])  # Capture all but auxiliary verbs
	        if 'are there' in question.lower():
	            # Type 1 & general "how many" questions
	            return f"Are there {answer} {subject}?"
	        elif 'are in' in question.lower():
	            # Type 2 questions specifically about being "in" something
	            return f"Are there {answer} {subject}?"
	        else:
	            # Type 3 questions that don't fit neatly into the above two categories
	            return f"Are there {answer} {subject}?"
	    elif start_phrase.lower().startswith('what number is'):
	        # Type 4 questions about identifying specific numbers
	        subject = ' '.join([token.text for token in doc[3:]])  # Exclude 'What number is'
	        return f"Is the number {answer} on {subject}?"
	    else:
	        # Fallback for unexpected formats
	        return f"Does the question '{question}' correspond to the number {answer}?"

	def transform_location_question(self, question, answer):
	    doc = nlp(question)
	    start_phrase = ' '.join([token.text for token in doc[:3]])  # Adjust slice as needed based on typical question length

	    if start_phrase.lower().startswith('where is the'):
	        # Type 1 questions: Single entity, singular
	        entity = ' '.join([token.text for token in doc[3:]])  # Extract entity after 'where is the'
	        return f"Is the {entity} in {answer}?"
	    elif start_phrase.lower().startswith('where are the'):
	        # Type 2 questions: Multiple entities, plural
	        entities = ' '.join([token.text for token in doc[3:]])  # Extract entities after 'where are the'
	        return f"Are the {entities} in {answer}?"
	    elif start_phrase.lower().startswith('what room is'):
	        # Type 3 questions: About a room, likely singular
	        return f"Is this the {answer} room?"
	    else:
	        # Fallback for less common or more complex starts
	        return f"Does the question '{question}' refer to {answer}?"	

	def transform_question_to_yes_no(self,question, answer):
		# Define common query words
		query_words = {'what', 'who'}
		
		# Parse the question using spaCy
		doc = nlp(question)
		new_question_parts = []

		# Collect parts of the question to keep, skipping query words
		for token in doc:
			if token.text.lower() not in query_words and token.dep_ != 'punct':
				new_question_parts.append(token.text)
		
		# Concatenate the retained parts with the answer
		if new_question_parts:
			# Start with 'Is' or 'Are' based on the plurality of the answer or context
			verb = 'Are' if answer.strip().split()[0].lower() in ['they', 'we'] else 'Is'
			new_question = f"{verb} {' '.join(new_question_parts)} {answer}?"
		else:
			# Fallback if no appropriate parts are found
			new_question = f"Is it {answer}?"

		new_question = new_question[0].upper() + new_question[1:]
		new_ques_words = new_question.split()
		if new_ques_words[0].lower() == new_ques_words[1]:
			new_ques_words.pop(1)
		elif new_ques_words[0].lower() == "Is" and new_ques_words[1] == 'are':
			new_ques_words.pop(0)
		new_question = " ".join(new_ques_words)
		return new_question


if __name__ == "__main__":
	path = "../datasets/vqa/Partition_Q_V2_subset_ST/"
	fname = "karpathy_train_q_judge.json"
	source = os.path.join(path, fname)
	ckpt_path='snap/blip_base/base.pth'
	estimator = YesNo(ckpt_path)
	with open(source, 'r') as f:
		data_dicts = json.load(f)
	rephrased = []
	for data in data_dicts:
		if "Q_q_recognition" in data and "A_self_q_recognition" in data:
			if data['A_self_q_recognition'] != 'not sure':
				img_id = data['img_id']
				split = "train" if 'train2014' in img_id else 'val'
				img_path = os.path.join(f"../datasets/COCO/{split}2014", f"{img_id}.jpg")
				question = data['Q_q_recognition']
				answer = data['A_self_q_recognition']
				query = estimator.transform_question_to_yes_no(question, answer)
				ans = estimator.get_yes_no(img_path, f"Question: {query} Answer:")
				import pdb;pdb.set_trace()
				rephrased.append(query)

	with open('temp_rec.txt', 'a') as f:
		for query in rephrased:
			f.write(f"{query}\n")


