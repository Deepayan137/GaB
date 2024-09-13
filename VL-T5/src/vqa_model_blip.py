from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import time
from transformers import AutoProcessor
from src.blip2.modeling_blip import NaiveBlip2VQACL
from src.blip2.modeling_blip_vqacl import Blip2VQACL
from src.blip2.modeling_blip_vqg import NaiveBlip2VQG
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
		
		attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
		cate_labels = batch['cate_labels'].to(device)
		ques_labels = batch['ques_labels'].to(device)
		
		max_new_tokens = 2
		output = self.generate(input_ids=input_ids,
			attention_mask=attention_mask,
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
		lambda_l2p=0.0,
		prompt_pool=False,
		use_cap_loss=False,
		ft_layers='query_tokens',):
		super().__init__(config, pool_size, prompt_pool, lambda_l2p)
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

		if ft_layers == 'query_tokens':
			print("Unfreeze only the query tokens")
			self.query_tokens.requires_grad = True
			self.language_projection_answers.requires_grad = True
			self.language_projection_questions.requires_grad = True
		
		print("Freeze Language model")
		for name, param in self.language_model.named_parameters():
			param.requires_grad = False

		self.count_parameters()
	
	def count_parameters(self):
		total_params = sum(p.numel() for p in self.parameters())
		trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		
		# Print trainable and total parameters
		print(f'Total parameters: {total_params}')
		print(f'Trainable parameters: {trainable_params}')
		
		# Count parameters for each module or parameter
		def count_module_params(module, module_name):
			if isinstance(module, nn.Module):
				total = sum(p.numel() for p in module.parameters())
				trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
			else:
				total = module.numel()
				trainable = module.numel() if module.requires_grad else 0
			
			print(f'{module_name} - Total: {total}, Trainable: {trainable}')
		
		count_module_params(self.vision_model, 'Vision Model')
		count_module_params(self.qformer, 'Query Transformer (Q-Former)')
		count_module_params(self.language_model, 'Language Model')
		count_module_params(self.language_projection_answers, 'Answer Projection Head')
		count_module_params(self.language_projection_questions, 'Question Projection Head')
		count_module_params(self.query_tokens, 'Query Tokens')

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
		if 'target_ids' in batch:
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

class NaiveBlipVQG(NaiveBlip2VQG):
	def __init__(self, config):
		super().__init__(config)
		for name, param in self.vision_model.named_parameters():
			param.requires_grad = False
		print("Freeze vision encoder")
		self.vision_model = self.vision_model.eval()
		# Freeze all parameters of the query transformer by default
		for param in self.qformer.parameters():
			param.requires_grad = False
		print("Unfreeze only the query tokens")
		self.query_tokens.requires_grad = True
		self.language_projection.requires_grad = True
		print("Freeze Language model")
		for name, param in self.language_model.named_parameters():
			param.requires_grad = False



	# @torch.no_grad()
	# def get_questions(self, batch, **kwargs):
	# 	self.eval()
	# 	max_new_tokens = kwargs['max_new_tokens']
	# 	device = next(self.parameters()).device
	# 	pixel_values = batch['pixel_values'].to(device)
	# 	attention_mask = None
	# 	if 'attention_mask' in batch:
	# 		attention_mask = batch['attention_mask']
	# 	input_ids = None
	# 	if 'input_ids' in batch:
	# 		input_ids = batch['input_ids'].to(device)
	# 		output = self.generate(
	# 			pixel_values,
	# 			'answers',
	# 			input_ids=input_ids,
	# 			attention_mask=attention_mask,
	# 			max_new_tokens=max_new_tokens, 
	# 			repetition_penalty=1.2)
	# 	else:
	# 		output = self.generate(
	# 			pixel_values,
	# 			'questions', 
	# 			max_new_tokens=max_new_tokens,
	# 			repetition_penalty=1.2)
	# 	result = {}
	# 	result['token_ids'] = output
	# 	result['questions'] = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True) 
	# 	return result

	# @torch.no_grad()
	# def test_step(self, batch, task, **kwargs):
	# 	self.eval()
	# 	device = next(self.parameters()).device
	# 	pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
	# 	input_ids = batch['input_ids'].to(device) # bs, 20
	# 	attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
	# 	max_new_tokens = 2
	# 	output = self.generate(
	# 		pixel_values,
	# 		input_ids,
	# 		'answers',
	# 		attention_mask=attention_mask,
	# 		max_new_tokens=max_new_tokens, 
	# 		repetition_penalty=1.2)
	# 	result = {}
	# 	result['token_ids'] = output
	# 	pred_ans = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True) 
	# 	if pred_ans == ['yesno']:
	# 		pred_ans = ['yes']
	# 	elif pred_ans == ['noyes']:
	# 		pred_ans = ['no']
	# 	elif pred_ans == ['yesyes']:
	# 		pred_ans = ['yes']
	# 	elif pred_ans == ['nono']:
	# 		pred_ans = ['no']
	# 	result['pred_ans'] = pred_ans
	# 	return result

	# def train_step(self, batch, current_task_id):
	# 	device = next(self.parameters()).device
	# 	pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
	# 	input_ids = batch['input_ids'].to(device) # bs, 20
	# 	attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
	# 	if attention_mask is None:
	# 		attention_mask = torch.ones_like(input_ids)
	# 	language_model_inputs, language_model_attention_mask, vision_outputs = self.encode_images(pixel_values, 'answers')
	# 	inputs_embeds = self.encode_answers(input_ids)
	# 	expected_device = language_model_attention_mask.device
	# 	attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
	# 	lm_labels = batch["target_ids"].to(device)
	# 	output = self.decode_answers(
	# 		inputs_embeds,
	# 		attention_mask,
	# 		lm_labels)
	# 	assert 'loss' in output
	# 	B, L = lm_labels.size()
	# 	loss = output['loss'] # 400 (bs*5)
	# 	result = {
	# 		'loss': loss
	# 	}
	# 	if 'cap_ids' in batch and self.use_cap_loss:
	# 		cap_ids = batch['cap_ids'].to(device)
	# 		cap_labels = cap_ids
	# 		attention_mask = (cap_ids != self.processor.tokenizer.pad_token_id).long().to(device)
	# 		language_model_inputs, language_model_attention_mask, vision_outputs = self.encode_images(pixel_values, 'questions')
	# 		inputs_embeds = self.encode_answers(cap_ids)
	# 		attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
	# 		mus, logvars = self.encode_into_z(language_model_inputs, inputs_embeds.to(language_model_inputs.device))
	# 		zs = self.reparameterize(mus, logvars)
	# 		inputs_embeds =  self.z_decoder(zs)
	# 		output_cap = self.decode_answers(inputs_embeds,attention_mask)
	# 		assert 'loss' in output_cap
	# 		loss_cap = output_cap['loss']
	# 		kl_loss = gaussian_KL_loss(mus, logvars)
	# 		result['loss_cap'] = loss_cap
		
	# 		#####
	# 		inputs_embeds = inputs_embeds.detach()
	# 		recon_image_features, recon_answer_features = vqg.reconstruct_inputs(
	# 				language_model_inputs, inputs_embeds)
			
	# 		recon_a_loss = l2_criterion(recon_answer_features, inputs_embeds)
	# 		total_recon_answer_loss += recon_a_loss.item()
	# 		#####


	# 	result['logits'] = output['logits']
	# 	result['BL'] = (B, L)
	# 	if 'loss_memory' in output:
	# 		result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
	# 	if 'loss_memory_new' in output:
	# 		result['loss_memory_new'] = output['loss_memory_new']
	# 	return result


if __name__ == "__main__":
	import torch
	import torch.nn as nn
	from transformers import T5Config, BartConfig, Blip2Config
	from torch.utils.data import DataLoader, Dataset
	from torch.optim.lr_scheduler import ReduceLROnPlateau
	from src.vqa_data_blip import VQADataset, VQAFineTuneDataset
	from src.param import parse_args
	# from src.vqacl import Trainer
	import sys
	from tqdm import *
	from Question_type import All_task, Category_splits
	import os
	backbone = "Salesforce/blip2-opt-2.7b"
	config = Blip2Config.from_pretrained(backbone)
	processor = AutoProcessor.from_pretrained(backbone)
	# model = Blip2ForConditionalGeneration.from_pretrained(backbone, config=config)
	model = NaiveBlipVQG.from_pretrained(backbone, config=config)
	task = 'q_recognition'
	save_path = f'snap/test/{task}_LAST.pth'
	device = 'cuda'
	model = model.to(device)
	# if os.path.exists(save_path):
	#     print(f'Loading model at {save_path}')
	#     ckpt = torch.load(save_path)
	#     model.load_state_dict(ckpt)
	split = 'train'
	coco_Ours = All_task
	train_dset = VQADataset(f"karpathy_{split}", True)
	val_dset = VQADataset(f"karpathy_val", True)
	args = parse_args()
	args.backbone = backbone
	args.use_class_hierarchy = False
	dataset = VQAFineTuneDataset(
				coco_Ours,
				[],
				'karpathy_train',
				raw_dataset=train_dset,
				rank=0,
				topk=-1,
				verbose=True,
				args=args,
				mode='train',
				task=task,
				cates=Category_splits['G1']
			)
	train_loader = DataLoader(
				dataset, batch_size=5, shuffle=True,
				num_workers=0, pin_memory=True, sampler=None,
				collate_fn=dataset.collate_fn)
	total_steps = len(train_loader)
	dataset = VQAFineTuneDataset(
				coco_Ours,
				[],
				f'karpathy_test',
				raw_dataset=val_dset,
				rank=0,
				topk=-1,
				verbose=True,
				args=args,
				mode='val',
				task=task,
				cates=[i for i in range(80)]
			)
	val_loader =  DataLoader(
				dataset, batch_size=2, shuffle=False,
				num_workers=0, pin_memory=True, sampler=None,
				collate_fn=dataset.collate_fn, drop_last=False)
	epoch_results = {
						'loss': 0.,
					}
	eos_token_id = processor.tokenizer('\n', add_special_tokens=False).input_ids[0]
	gen_params = model.generator_parameters()
	info_params =model.info_parameters()
	l2_criterion = nn.MSELoss()
	gen_optimizer = torch.optim.Adam(gen_params, lr=1e-4)
	info_optimizer = torch.optim.Adam(info_params, lr=1e-4)
	scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
								  factor=0.1, patience=10,
								  verbose=True, min_lr=1e-7)
	info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
									   factor=0.1, patience=10,
									   verbose=True, min_lr=1e-7)
	num_epochs = 1
	# trainer = Trainer(args, All_task)

	def run_eval(args, model, data_loader, l2_criterion, epoch, scheduler, info_scheduler):
		print('=' * 80)
		start_time = time.time()
		val_gen_loss, val_info_loss = evaluate(args, model, data_loader, l2_criterion)
		delta_time = time.time() - start_time
		scheduler.step(val_gen_loss)
		scheduler.step(val_info_loss)
		print(f"Time: {delta_time:.4f}, Epoch [{epoch}/{args.num_epochs}], Val-gen-loss: {val_gen_loss:.4f}, Val-info-loss: {val_info_loss:.4f}")
		print('=' * 80)

	def evaluate(args, model, data_loader, l2_criterion):
		preds = []
		truths = []
		print("Validating")
		for batch in tqdm(data_loader):  # Changed loader to data_loader for consistency
			pixel_values = batch['pixel_values'].to(device)  # bs, 36, 2048
			input_ids = batch['input_ids'].to(device)  # bs, 20
			lm_labels = batch["target_ids"].to(device)
			attention_mask = (lm_labels != processor.tokenizer.pad_token_id).long().to(device)
			output = model.generate(input_ids=lm_labels,
									pixel_values=pixel_values,
									attention_mask=attention_mask, max_new_tokens=20)
			out_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
			preds.append(out_text[0])
			truths.append(batch['answers'][0])
			import pdb;pdb.set_trace()
		# acc = trainer.evaluate(data_loader)  # This line may need correction based on the actual method signature
		# print(f"Accuracy: {acc}")
	

	
	print("Training starts:")
	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader_cate):
			pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
			input_ids = batch['input_ids'].to(device) # bs, 20
			lm_labels = batch["target_ids"].to(device)
			gen_optimizer.zero_grad()
			info_optimizer.zero_grad()
			language_model_inputs, language_model_attention_mask, vision_outputs, query_outputs = model.encode_images(pixel_values, 'answers')
			answer_features = model.encode_answers(lm_labels)
			answer_features = torch.cat([language_model_inputs, answer_features.to(language_model_inputs.device)], dim=1)
			mus, logvars = model.encode_into_z(language_model_inputs, answer_features)
			zs = model.reparameterize(mus, logvars)
			answer_attention_mask = (lm_labels != processor.tokenizer.pad_token_id).long().to(device)
			answer_attention_mask = torch.cat([language_model_attention_mask, answer_attention_mask.to(device)], dim=1)
			output = model.decode_questions(answer_features, vision_outputs, query_outputs, answer_attention_mask, input_ids)
			gen_loss = output["loss"]
			total_loss = 0.0
			total_loss += gen_loss
			gen_loss = gen_loss.item()
			kl_loss = model.gaussian_KL_loss(mus, logvars)
			total_loss += 0.001*kl_loss
			kl_loss = kl_loss.item()
			
			total_loss.backward()
			gen_optimizer.step()
			total_info_loss = 0.0
			gen_optimizer.zero_grad()
			info_optimizer.zero_grad()
			recon_answer_loss = 0.0
			answer_targets = answer_features.detach()
			image_targets = language_model_inputs.detach()
			recon_image_features, recon_answer_features = model.reconstruct_inputs(
				image_targets, answer_targets)
			recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
			total_info_loss += 0.001 * recon_a_loss
			recon_answer_loss = recon_a_loss.item()
			
			# optimizer.zero_grad()
			total_info_loss.backward()
			info_optimizer.step()
		
		if i % 10 == 0:
			delta_time = time.time() - start_time
			start_time = time.time()
			logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
					 'LR: %f, gen: %.4f, KL: %.4f, '
					 'A-recon: %.4f,'
					 
					 % (delta_time, epoch, num_epochs, i,
						total_steps, gen_optimizer.param_groups[0]['lr'],
						gen_loss, kl_loss,
						recon_answer_loss,
						))
	torch.save(model.state_dict(), os.path.join('snap/test/q_location.pth'))
	run_eval(args, model, val_loader, l2_criterion,
			 epoch, scheduler, info_scheduler)
		
