import torch
import torch.nn as nn
from transformers import T5Config, BartConfig, Blip2Config
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset
from src.vqa_model_blip import NaiveBlipVQG
from src.param import parse_args
import sys
from tqdm import *
from Question_type import All_task, Sg_task, Category_splits
import os

device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
	backbone = "Salesforce/blip2-opt-2.7b"
	config = Blip2Config.from_pretrained(backbone)
	processor = AutoProcessor.from_pretrained(backbone)
	model = NaiveBlipVQG.from_pretrained(backbone, config=config)
	save_path = f'snap/test/{task}_LAST.pth'
	model = model.to(device)
	return model, processor

def get_train_val_loaders(dataset_name, task):
	if dataset_name == 'vqacl':
		coco_Ours = All_task
		from src.vqa_data_blip import get_loader, get_loader_test
		train_dset = VQADataset(f"karpathy_{split}", True)
		train_loader, _ = get_loader(args, 
			coco_Ours, 
			[], 
			train_dset, 
			split='karpathy_train', 
			mode='train', batch_size=32,
			task=task)
		test_dset = VQADataset(f"karpathy_test", True)
		test_loader = get_loader_test(
						args,
						coco_Ours,
						[],
						test_dset,
						split='karpathy_train', 
						mode='val', batch_size=32,
						workers=4,
						task=task,
					)

		return train_loader['G1'], test_loader['G1']
	elif dataset_name == 'sgvqa':
		from src.sgvqa_data_blip import get_loader, get_loader_test
		train_loader, _ = get_loader(args,
			split='train', 
			scenario='function', 
			batch_size=32,
			workers=4,task='object')
		test_loader = get_loader_test(args,
			split='val',
			scenario='function', 
			batch_size=32,
			workers=4,task='object')

		return train_loader, test_loader
	raise ValueError("Wrong dataset")

def run_eval(args, model, processor, data_loader, l2_criterion, epoch, scheduler, info_scheduler, task):
	print('=' * 80)
	start_time = time.time()
	val_gen_loss, val_info_loss = evaluate(args, model, processor, data_loader, l2_criterion)
	delta_time = time.time() - start_time
	scheduler.step(val_gen_loss)
	scheduler.step(val_info_loss)
	print(f"Time: {delta_time:.4f}, Epoch [{epoch}/{args.num_epochs}], Val-gen-loss: {val_gen_loss:.4f}, Val-info-loss: {val_info_loss:.4f}")
	print('=' * 80)

def evaluate(args, model, processor, data_loader, l2_criterion):
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

def train_model(args, model, processor, gen_optimizer, info_optimizer, l2_criterion, train_loader, num_epochs, task):
	total_steps = len(train_loader)
	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader):
			pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
			input_ids = batch['input_ids'].to(device) # bs, 20
			lm_labels = batch["target_ids"].to(device)
			gen_optimizer.zero_grad()
			info_optimizer.zero_grad()
			language_model_inputs, language_model_attention_mask, vision_outputs, query_outputs = model.encode_images(pixel_values)
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
			recon_image_features, recon_answer_features = model.reconstruct_inputs(image_targets, answer_targets)
			recon_a_loss = l2_criterion(recon_answer_features, answer_targets)
			total_info_loss += 0.001 * recon_a_loss
			recon_answer_loss = recon_a_loss.item()
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
	torch.save(model.state_dict(), os.path.join(f'vqg_model/{task}.clpt'))

if __name__ == "__main__":
	args = parse_args()
	args.use_class_hierarchy = False
	DATASET='sgvqa'
	task='object'
	train_loader, test_loader = get_train_val_loaders(DATASET, task)
	model, processor = get_model()
	gen_params = model.generator_parameters()
	info_params =model.info_parameters()
	gen_optimizer = torch.optim.Adam(gen_params, lr=1e-4)
	info_optimizer = torch.optim.Adam(info_params, lr=1e-4)
	scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
								  factor=0.1, patience=10,
								  verbose=True, min_lr=1e-7)
	info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
									   factor=0.1, patience=10,
									   verbose=True, min_lr=1e-7)
	l2_criterion = nn.MSELoss()
	num_epochs = 1
	# trainer = Trainer(args, All_task)
	print("Training starts:")
	train_model(args, model, processor, gen_optimizer, info_optimizer, l2_criterion, train_loader, num_epochs, task)
	run_eval(args, model, processor, val_loader, l2_criterion,
			 epoch, scheduler, info_scheduler)
