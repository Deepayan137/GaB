from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from transformers import AutoProcessor
from src.blip2.modeling_blip import NaiveBlip2VQACL
from src.blip2.modeling_blip_vqacl import Blip2VQACL

class BLIP2Prototype(Blip2VQACL):
	def __init__(self, config, num_answers=None, label2ans=None, ft_layers="query_tokens"):
		super().__init__(config)
		self.num_answers = num_answers
		self.label2ans = label2ans
		self.bce_loss = nn.BCEWithLogitsLoss()
		self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		# self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
		for name, param in self.vision_model.named_parameters():
			param.requires_grad = False
		print("Freeze vision encoder")
		self.vision_model = self.vision_model.eval()

		num_layers = len(self.qformer.encoder.layer)
		# Freeze all parameters of the query transformer by default
		for param in self.qformer.parameters():
			param.requires_grad = False

		print("Freeze Language model")
		for name, param in self.language_model.named_parameters():
			param.requires_grad = False

	def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False):
		device = next(self.parameters()).device
		pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
		input_ids = batch['input_ids'].to(device) # bs, 20
		lm_labels = batch["target_ids"].to(device) #[bs, 5]
		cate_labels = batch['cate_labels'].to(device)
		ques_labels = batch['ques_labels'].to(device)
		# attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)

		output = self(
			input_ids=input_ids,
			pixel_values=pixel_values,
			attention_mask=None,
			labels=lm_labels,
			cate_labels=cate_labels,
			ques_labels=ques_labels,
			proto_update=True,
			memory=memory,
			current_task_id=current_task_id,
			mem_num_Q = mem_num_Q,
			total_num_Q = total_num_Q,
			proto_alpha=proto_alpha,
			proto_beta=proto_beta,
			return_dict=True
		)

		assert 'loss' in output
		lm_mask = (lm_labels != 1).float()
		B, L = lm_labels.size()
		loss = output['loss'].mean() # 400 (bs*5)
		result = {
			'loss': loss
		}
		result['logits'] = output['logits']
		result['BL'] = (B, L)
		if 'loss_memory' in output:
			result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
		if 'loss_memory_new' in output:
			result['loss_memory_new'] = output['loss_memory_new']
		return result
		
	@torch.no_grad()
	def test_step(self, batch, task, **kwargs):
		self.eval()
		device = next(self.parameters()).device
		pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
		input_ids = batch['input_ids'].to(device) # bs, 20
		lm_labels = batch["target_ids"].to(device) #[bs, 5]
		
		# attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
		cate_labels = batch['cate_labels'].to(device)
		ques_labels = batch['ques_labels'].to(device)
		if task in ['q_judge', 'q_commonsense']:
			max_new_tokens = 1
		else:
			max_new_tokens = 2
		output = self.generate(
			input_ids=input_ids,
			pixel_values=pixel_values,
			max_new_tokens=max_new_tokens,
			repetition_penalty=1.2)
		result = {}
		result['token_ids'] = output
		result['pred_ans'] = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)
		return result


class NaiveBLIP2(NaiveBlip2VQACL):
	def __init__(self, config, 
		num_answers=None, 
		label2ans=None, 
		pool_size=None,
		prompt_pool=False,
		use_cap_loss=False,
		ft_layers='query_tokens',):
		super().__init__(config, pool_size, prompt_pool)
		from transformers import AutoProcessor
		self.use_cap_loss = use_cap_loss
		self.num_answers = num_answers
		self.label2ans = label2ans
		self.bce_loss = nn.BCEWithLogitsLoss()
		self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		for name, param in self.vision_model.named_parameters():
			param.requires_grad = False
		print("Freeze vision encoder")
		self.vision_model = self.vision_model.eval()
		num_layers = len(self.qformer.encoder.layer)
		# Freeze all parameters of the query transformer by default
		for param in self.qformer.parameters():
			param.requires_grad = False

		if ft_layers == 'full':
			print("Unfreeze all parameters of the query transformer")
			for param in self.qformer.parameters():
				param.requires_grad = True

		elif ft_layers == 'query_tokens':
			print("Unfreeze only the query tokens")
			self.query_tokens.requires_grad = True
			self.language_projection_answers.requires_grad = True
			self.language_projection_questions.requires_grad = True
		
		elif ft_layers == 'query_tokens_random':
			print("Unfreeze only the query tokens")
			self.query_tokens.requires_grad = True
			in_features, out_features = self.language_projection.in_features,\
			self.language_projection.out_features
			self.language_projection = nn.Linear(in_features, out_features)
			self.language_projection.weight.requires_grad = True
			self.language_projection.bias.requires_grad = True
		elif ft_layers == 'query_tokens_task':
			num_heads = 10
			self.query_tokens.requires_grad = True
			in_features, out_features = self.language_projection.in_features,\
			self.language_projection.out_features
			self.projection_heads = nn.ModuleList([
					nn.Linear(in_features, out_features) for _ in range(num_heads)
				])
			for head in self.projection_heads:
				init.xavier_uniform_(head.weight)  # Xavier initialization
				init.zeros_(head.bias)

			self.meta_selector = nn.Linear(out_features, num_heads)
			init.xavier_uniform_(self.meta_selector.weight)  # Xavier initialization
			init.zeros_(self.meta_selector.bias)

		print("Freeze Language model")
		for name, param in self.language_model.named_parameters():
			param.requires_grad = False
		
	@torch.no_grad()
	def get_questions(self, batch, **kwargs):
		self.eval()
		max_new_tokens = kwargs['max_new_tokens']

		device = next(self.parameters()).device
		pixel_values = batch['pixel_values'].to(device)
		attention_mask = None
		if 'attention_mask' in batch:
			attention_mask = batch['attention_mask']
		query_outputs, vision_outputs = self.get_features(pixel_values)
		input_ids = None
		if 'input_ids' in batch:
			input_ids = batch['input_ids'].to(device)
			output = self.generate(query_outputs=query_outputs, vision_outputs=vision_outputs, max_new_tokens=max_new_tokens,attention_mask=attention_mask,input_ids=input_ids, repetition_penalty=1.2,mode='questions')
		else:
			output = self.generate(query_outputs=query_outputs, vision_outputs=vision_outputs, max_new_tokens=max_new_tokens,repetition_penalty=1.2, mode='questions')
		result = {}
		result['token_ids'] = output
		result['questions'] = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True) 
		return result

	@torch.no_grad()
	def test_step(self, batch, task, **kwargs):
		self.eval()
		device = next(self.parameters()).device
		pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
		query_outputs, vision_outputs = self.get_features(pixel_values)
		input_ids = batch['input_ids'].to(device) # bs, 20
		lm_labels = batch["target_ids"].to(device) #[bs, 5]
		
		attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
		# cate_labels = batch['cate_labels'].to(device)
		# ques_labels = batch['ques_labels'].to(device)
		max_new_tokens = 2
		output = self.generate(
			query_outputs=query_outputs, 
			vision_outputs=vision_outputs, 
			input_ids=input_ids, 
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens, 
			repetition_penalty=1.2,
			mode='answers')
		result = {}
		result['token_ids'] = output
		pred_ans = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True) 
		if pred_ans == ['yesno']:
			pred_ans = ['yes']
		elif pred_ans == ['noyes']:
			pred_ans = ['no']
		elif pred_ans == ['yesyes']:
			pred_ans = ['yes']
		elif pred_ans == ['nono']:
			pred_ans = ['no']
		result['pred_ans'] = pred_ans
		return result

	def train_step(self, batch, current_task_id):
		device = next(self.parameters()).device
		pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
		query_outputs, vision_outputs = self.get_features(pixel_values)
		input_ids = batch['input_ids'].to(device) # bs, 20
		lm_labels = batch["target_ids"].to(device) #[bs, 5]
		attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
		
		output = self(
			query_outputs=query_outputs,
			vision_outputs=vision_outputs,
			input_ids=input_ids,
			attention_mask=attention_mask,
			labels=lm_labels,
			mode='answers')
		assert 'loss' in output
		B, L = lm_labels.size()
		loss = output['loss'] # 400 (bs*5)
		result = {
			'loss': loss
		}
		if 'cap_ids' in batch and self.use_cap_loss:
			cap_ids = batch['cap_ids'].to(device)
			cap_labels = cap_ids
			output_cap = self(
				query_outputs=query_outputs,
				vision_outputs=vision_outputs,
				input_ids=cap_ids,
				labels=cap_labels,
				mode='questions')

			assert 'loss' in output_cap
			loss_cap = output_cap['loss']
			result['loss_cap'] = loss_cap
		result['logits'] = output['logits']
		result['BL'] = (B, L)
		if 'loss_memory' in output:
			result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
		if 'loss_memory_new' in output:
			result['loss_memory_new'] = output['loss_memory_new']
		return result


# if __name__ == "__main__":
#     import torch
#     import torch.nn as nn
#     from transformers import T5Config, BartConfig, Blip2Config
#     from torch.utils.data import DataLoader, Dataset
#     from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
#     from src.param import parse_args
#     # from src.vqacl import Trainer
#     import sys
#     from tqdm import *
#     from Question_type import All_task, Category_splits
#     import os
#     backbone = "Salesforce/blip2-opt-2.7b"
#     config = Blip2Config.from_pretrained(backbone)
#     processor = AutoProcessor.from_pretrained(backbone)
#     # model = Blip2ForConditionalGeneration.from_pretrained(backbone, config=config)
#     model = BLIP2Prototype.from_pretrained(backbone, config=config)
#     task = 'q_recognition'
#     save_path = f'snap/test/{task}_LAST.pth'
#     device = 'cuda'
#     model = model.to(device)
#     # if os.path.exists(save_path):
#     #     print(f'Loading model at {save_path}')
#     #     ckpt = torch.load(save_path)
#     #     model.load_state_dict(ckpt)
#     split = 'train'
#     coco_Ours = All_task
#     train_dset = VQADataset(f"karpathy_{split}", True)
#     val_dset = VQADataset(f"karpathy_val", True)
#     args = parse_args()
#     args.backbone = backbone

#     dataset = VQAFineTuneDataset(
#                 coco_Ours,
#                 [],
#                 'karpathy_train',
#                 raw_dataset=train_dset,
#                 rank=0,
#                 topk=-1,
#                 verbose=True,
#                 args=args,
#                 mode='train',
#                 task=task,
#                 cates=Category_splits['G1']
#             )
#     train_loader_cate = DataLoader(
#                 dataset, batch_size=5, shuffle=True,
#                 num_workers=0, pin_memory=True, sampler=None,
#                 collate_fn=dataset.collate_fn)
#     dataset = VQAFineTuneDataset(
#                 coco_Ours,
#                 [],
#                 f'karpathy_test',
#                 raw_dataset=val_dset,
#                 rank=0,
#                 topk=-1,
#                 verbose=True,
#                 args=args,
#                 mode='val',
#                 task=task,
#                 cates=[i for i in range(80)]
#             )
#     val_loader_cate =  DataLoader(
#                 dataset, batch_size=2, shuffle=False,
#                 num_workers=0, pin_memory=True, sampler=None,
#                 collate_fn=dataset.collate_fn, drop_last=False)
#     epoch_results = {
#                         'loss': 0.,
#                     }
#     eos_token_id = processor.tokenizer('\n', add_special_tokens=False).input_ids[0]
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
#     num_epochs = 1
#     # trainer = Trainer(args, All_task)
#     def preprocess(text):
#         # Convert to lowercase, strip whitespace, remove punctuation, etc.
#         text = text.lower().strip()
#         return text
	
#     def evaluate(predictions, truths):
#         total = len(predictions)
#         correct = 0

#         for pred, truth in zip(predictions, truths):
#             if preprocess(pred) in preprocess(truth) or preprocess(truth) in preprocess(pred):
#                 correct += 1
#         accuracy = correct / total
#         return accuracy

#     def validation(model, loader):
#         # preds = []
#         # truths = []
#         # print("Validating")
#         # for batch in tqdm(loader):
#         #     pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
#         #     input_ids = batch['input_ids'].to(device) # bs, 20
#         #     lm_labels = batch["target_ids"].to(device)
#         #     attention_mask = (input_ids != processor.tokenizer.pad_token_id).long().to(device)
#         #     output = model.generate(input_ids=input_ids,
#         #         pixel_values=pixel_values,
#         #         attention_mask=attention_mask, max_new_tokens=20)
#         #     out_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
#         #     preds.append(out_text[0])
#         #     truths.append(batch['answers'][0])
#         acc = trainer.evaluate(loader)
#         print(f"Accuracy:{acc}")
	
#     # validation(model, val_loader_cate)
#     print("Training starts:")
#     # for epoch in range(num_epochs):
#     # for task in All_task[1:]:
#     #     with tqdm(total=len(train_loader_cate), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
#     #         for batch in tqdm(train_loader_cate):
#                 # pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
#                 # input_ids = batch['input_ids'].to(device) # bs, 20
#                 # lm_labels = batch["target_ids"].to(device)
#                 # attention_mask = (input_ids != processor.tokenizer.pad_token_id).long().to(device)
#                 # output = model.train_step(batch, 0, args.proto_alpha, args.proto_beta)
#     #             output = model.test_step(batch, 0)
#     #             loss = output["loss"]
#     #             loss.backward()
#     #             optimizer.step()
#     #             optimizer.zero_grad()
#     #             pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss:.4f}')
#     #             pbar.update(1)
#     #         validation(model, val_loader_cate)
#     # torch.save(model.state_dict(), os.path.join('snap/test/q_location.pth'))
