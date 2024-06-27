import os
import sys
import json
import random
import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor
import collections
from pathlib import Path
from packaging import version
import copy
import numpy as np
from tqdm import tqdm
import pickle

import logging
import shutil
from pprint import pprint

from param import parse_args
from utils import load_state_dict, LossMeter, set_global_logging_level

from trainer_base import TrainerBase
from src.data_utils_sgvqa import create_rehearsal_data
sys.path.insert(0, '../')
from Question_type import Sg_task, show_results_matrix

def cycle(iterable):
	# iterate with shuffling
	while True:
		for i in iterable:
			yield i



class Trainer(TrainerBase):
	def __init__(self, args, sg_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
		self.result_matrix = {}
		self.task_list = []
		for task in sg_Ours:
			self.result_matrix[task] = {}
			self.task_list.append(task)
		self.train_loader_dict = {}
		self.val_loader_dict = {}
		self.test_loader_dict = {}
		self.test_loader_dict_all = {}
		if 'blip' in args.backbone:
			from vqa_data_blip import VQADataset
			self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		elif 't5' in args.backbone:
		   from vqa_data_memory import VQADataset 
		super().__init__(
			args,
			train=train)
		from vqa_model import VLT5VQA
		from vqa_model_blip import NaiveBLIP2
		model_kwargs = {'ft_layers':args.ft_layers, 
		'pool_size':args.pool_size, 'prompt_pool':args.prompt_pool}
		if args.prompt_pool:
			print("Activating Learning to Prompt")
		# model_kwargs = {}
		if 't5' in args.backbone:
			model_class = VLT5VQA
		elif 'blip' in args.backbone:
			from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
			model_class = NaiveBLIP2
		config = self.create_config()
		self.tokenizer = self.create_tokenizer()
		self.model = self.create_model(model_class, config, **model_kwargs)
		self.regularizer = None
		if isinstance(self.model, tuple):
			self.model, self.regularizer = self.model
		if 't5' in self.args.tokenizer:
			self.model.resize_token_embeddings(self.tokenizer.vocab_size)
		elif 'bart' in self.args.tokenizer:
			self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)
		self.model.tokenizer = self.tokenizer
		self.start_epoch = None
		# GPU Options
		print(f'Model Launching at GPU {self.args.gpu}')
		if self.verbose:
			from time import time
			start = time()
		self.model = self.model.to(args.gpu)
		if self.regularizer is not None:
			self.regularizer = self.regularizer.to(args.gpu)
		if 'blip' in self.args.backbone:
			self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
			if self.args.use_cap_loss:
				print("We will use caption loss and two optimizers")
				self.optim_question = torch.optim.AdamW(params=self.model.language_projection_questions.parameters(),lr=1e-4,  
						weight_decay=self.args.warmup_ratio) # Using same weight decay as an example
			self.lr_scheduler = None
		if self.verbose:
			print(f'It took {time() - start:.1f}s')
		self.iftrain = train
		self.sg_Ours = sg_Ours
		self.task_iftrain = {}
		for task in self.sg_Ours:
			self.task_iftrain[task] = 0

		self.task_total_num = torch.zeros(len(self.task_list))
		self.M = args.m_size
		self.Examplar_set = {}
	
	def _load_checkpoint(self, checkpoint_name, latest_task_idx):
		checkpoint_model = f'{self.args.output}/{checkpoint_name}_BEST'
		for idx, task in enumerate(self.task_list):
			if idx <= latest_task_idx:
				self.task_iftrain[task] = 1
		self.load(checkpoint_model)
		print(f'Success to load the checkpoint from the task {checkpoint_name}')

	def build_data_info_path(self, scenario_dir, tsk):
		# Define the suffix based on M
		suffix_mapping = {
		    5000: '_5k',
		    1000: '_1k',
		    2500: '_2k',
		    10000: '_10k',
		    20000: '_20k'
		}

		# Determine the balance type
		if self.args.balance_strategy == "classifier":
			balance_type = "balanced"
		elif self.args.balance_strategy == "cluster":
			balance_type = "cluster_balanced"
		else:
			balance_type = "unbalanced"

		# Get the appropriate suffix for the given M, default to an empty string if not found
		suffix = suffix_mapping.get(self.M, '')

		# Construct the file path
		file_name = f"fcl_mmf_{tsk}_train_{balance_type}{suffix}.json"
		data_info_path = os.path.join(scenario_dir, file_name)

		return data_info_path

	def train(self, load=False):
		if 'blip' in args.backbone:
			from sgvqa_data_blip import get_loader, get_loader_test, get_loader_memory
		elif 't5' in args.backbone:
			from vqa_data_memory import get_loader, get_loader_test, get_loader_memory
		latest_task_idx = -1
		if load:
			latest_task = '_'.join(os.path.basename(self.args.checkpoint).split('_')[:1])
			latest_task_idx = self.task_list.index(latest_task)
			self.load(self.args.checkpoint)
			# self._load_checkpoint(latest_task, latest_task_idx)
		# run = wandb.init(
		#     # Set the project where this run will be logged
		#     project="scenevqa_15",
		#     # Track hyperparameters and run metadata
		#     config=vars(args)
		# )
		task2id = {self.task_list[i]:i for i in range(len(self.task_list))}
		id2task = {v:k for k, v in task2id.items()}
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if args.train_from_scratch:
			base_state = {
				"model":self.model.state_dict()
			}
			base_state_dict = {k: v.cpu() for k, v in base_state["model"].items()}
		for i, task in enumerate(self.task_list[latest_task_idx+1:]):
			if i > 0 and args.train_from_scratch:
				self.model.load_state_dict({k: v.to(device) for k, v in base_state_dict.items()})
				self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
				print("Reset model to base state and starting training for new task")
			task_idx = task2id[task]
			print('======================== Now is task "', task, '" ========================')
			self.task_iftrain[task] = 1
			# Memory
			if args.memory:
				if task_idx > 0:
					prev_task = self.task_list[task_idx - 1]
					ckpt_path = os.path.join(self.args.output, f'{prev_task}_BEST.pth')
					if os.path.exists(ckpt_path) and ckpt_path != self.args.checkpoint:
						self.load(ckpt_path) # Load the best previous model
					each_memory = int(self.M / task_idx)
					if not args.use_gen_data:
						if args.use_biased_data:
							print("Loading use_biased_data... Hahahahha")
							with open(f'../datasets/npy_biased/{task}.pkl', 'rb') as f:
								self.Examplar_set = pickle.load(f)
						else:
							for t in range(task_idx):
								tsk = self.task_list[t]
								if tsk not in self.Examplar_set.keys():
									data_info_path = (f'../datasets/npy/{args.scenario}/fcl_mmf_' + f'{tsk}_train.npy')
									data_info_dicts = np.load(data_info_path, allow_pickle=True)
									random.shuffle(data_info_dicts)  # shuffle
									self.Examplar_set[tsk] = data_info_dicts[:each_memory]
						All_examplar = []
						for task_set in self.Examplar_set:
							All_examplar.extend(self.Examplar_set[task_set][:each_memory])
					else:
						# Construct the task-specific file path
						# if self.args.replay_strategy == 'static':
						print("Welcome to the static rehearsal module")
						tsk = Sg_task[f'{args.scenario}'][args.sequence][task_idx]
						scenario_dir = f'../datasets/npy_no_ents/{args.scenario}'
						data_info_path = self.build_data_info_path(scenario_dir, tsk)
						print(f"Load synthetic replay data from {data_info_path}")
						# Load the exemplar data from the file
						with open(data_info_path, 'r') as file:
							All_examplar = json.load(file)
						# elif self.args.replay_type == "dynamic":
						# 	pass
					print(f"Size of Repay data is {len(All_examplar)}")
				else:
					All_examplar = []
					each_memory = 0
			else:
				All_examplar = []
				each_memory = 0

			# Load the data
			print("#Loading ", task)
			print(f'Building train loader at GPU {args.gpu}')
			if task != "scenetext":
				batch_size = args.batch_size
			else:
				batch_size = 1
			train_loader, total_num_Q = get_loader(
				args,
				split=args.train, scenario=args.scenario, 
				batch_size=batch_size,
				workers=args.num_workers,
				task=task,
			)

			self.task_total_num[task_idx] = total_num_Q

			print(f'Building val loader at GPU {args.gpu}')
			val_loader, _ = get_loader(
				args,
				split=args.valid, scenario=args.scenario, 
				batch_size=args.valid_batch_size,
				workers=4,
				task=task,
			)

			print(f'Building test loader at GPU {args.gpu}')
			test_loader = get_loader_test(
				args,
				split=args.test, scenario=args.scenario, 
				batch_size=args.valid_batch_size,
				workers=4,
				task=task,
			)
			self.test_loader_dict_all[task] = test_loader

			print("#Loading ", task)
			memory_loader = get_loader_memory(
				args,
				All_examplar,
				split=args.train, scenario=args.scenario, batch_size=32,
				workers=args.num_workers,
			)  #G1-G5

			# if self.verbose:
			loss_meter = LossMeter()
			loss_meter_mem = LossMeter()
			loss_meter_ques = LossMeter()
			loss_meter_reg = LossMeter()
			best_valid = 0.
			best_epoch = 0

			self.train_loader = train_loader
			self.val_loader = val_loader
			self.memory_loader = memory_loader
			# Optimizer
			if self.iftrain:
				if len(self.memory_loader.dataset) > 0:
					total_train_num = 2 * len(self.train_loader.dataset)
				else:
					total_train_num = len(self.train_loader.dataset)
				if 't5' in self.args.backbone:
					self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(total_train_num)
			self.scaler = None
			start_epoch = 0
			# score_dict = self.evaluate(self.val_loader_cate, task)
			valid_score_raw_best = 0.0
			patience_counter = 0
			patience = 5
			for epoch in range(start_epoch, self.args.epochs):
				if self.start_epoch is not None:
					epoch += self.start_epoch
				self.model.train()
				if args.show_train_progress:
					pbar = tqdm(total=len(self.train_loader), ncols=120)
				epoch_results = {
					'loss': 0.,
				}

				quesid2ans = {}
				if len(self.memory_loader.dataset) > 0:
					now_loader = zip(self.train_loader, cycle(self.memory_loader))
					print('Use memory loader')
				else:
					now_loader = self.train_loader

				for now_batch in now_loader:
					if len(now_batch) == 2:
						batch, mem_batch = now_batch
					else:
						batch = now_batch
						mem_batch = None
					results, lr = self.train_step(batch, epoch_results, task_idx, each_memory)
					if mem_batch:
						results_mem, lr = self.train_step(mem_batch, epoch_results, task_idx, each_memory)    
					loss_meter.update(results['loss'].item())
					if 'loss_cap' in results:
						loss_meter_ques.update(results['loss_cap'].item())
					else:
						loss_meter_ques.update(-1)
					desc_str = f'Epoch {epoch} | LR {lr:.6f} | Loss {loss_meter.val:.4f} |'
					if mem_batch:
						loss_meter_mem.update(results_mem['loss'].item())
						desc_str += f' | Loss_mem {loss_meter_mem.val:4f}'
					else:
						loss_meter_mem.update(-1)
					
					if 'reg_loss' in results:
						loss_meter_reg.update(results['reg_loss'].item())
						desc_str += f' | Loss_reg {loss_meter_reg.val:4f}'
					
					if args.show_train_progress:
						pbar.set_description(desc_str)
						pbar.update(1)

				if args.show_train_progress:
					pbar.close()
				print(f"Epoch {epoch}| Loss: {loss_meter.val}, Loss_mem: {loss_meter_mem.val}, Loss_Ques: {loss_meter_ques.val}")
				score_dict = self.evaluate(self.val_loader, task)
				valid_score_raw = score_dict['overall']
				# wandb.log({
				#   f"val_accuracy_{task}": valid_score_raw, 
				#   f"train_loss_{task}": loss_meter.val})
				log_str = ''
				log_str += "\nValid Raw %0.2f" % (valid_score_raw)
				print(log_str)
				if valid_score_raw > valid_score_raw_best:
					valid_score_raw_best = valid_score_raw
					patience_counter = 0  # Reset the patience counter
					print("Saving Best")
					self.save(task + "_BEST")
				else:
					patience_counter += 1  # Increment the patience counter
					print(f"No improvement for {patience_counter} epochs.")
				self.save(task + "_LAST")
				if patience_counter > patience:
					print("Early stopping triggered.")
					print("Saving Last")
					break  # Break out of the training loop
			
			if self.regularizer is not None:
				self.regularizer.after_training_exp(model=self.model,optimizer=self.optim,dloader=self.train_loader,current_task_id=task_idx)
		print("Saving Last")
		self.save(task + "_LAST")

	def train_step(self, batch, epoch_results, task_idx, each_memory):
		self.optim.zero_grad(set_to_none=True)
		results = self.model.train_step(batch, task_idx)
		loss = results['loss']
		if 'loss_cap' in results:
			if self.args.use_cap_loss:
				loss_cap = results['loss_cap']
				self.optim_question.zero_grad(set_to_none=True)
				loss_cap.backward(retain_graph=True)
				self.optim_question.step()
	    
        if self.regularizer is not None:
			cl_reg = self.regularizer.before_backward(self.model, device=loss.device)
			loss += cl_reg
			results['reg_loss'] = cl_reg
		
        loss.backward()
		loss = loss.detach()
		if self.args.clip_grad_norm > 0:
			 torch.nn.utils.clip_grad_norm_(
					self.model.parameters(), self.args.clip_grad_norm)
		self.optim.step()
		# 
		if self.lr_scheduler:
			self.lr_scheduler.step()
		for param in self.model.parameters():
			param.grad = None
		for k, v in results.items():
			if k in epoch_results:
				epoch_results[k] += v.item()

		if self.lr_scheduler:
			if version.parse(torch.__version__) >= version.parse("1.4"):
				lr = self.lr_scheduler.get_last_lr()[0]
			else:
				lr = self.lr_scheduler.get_lr()[0]
		else:
			try:
				lr = self.optim.get_lr()[0]
			except AttributeError:
				lr = self.args.lr
		return results, lr

	def Test(self, load=False):
		for task_idx, task in enumerate(self.task_list):
			print('======================== Now is task "', task, '" ========================')
			if 'blip' in self.args.backbone:
				from sgvqa_data_blip import get_loader_test, get_loader
			test_loader  = get_loader_test(
				args,
				split=args.test, 
				scenario=args.scenario, batch_size=args.valid_batch_size,
				workers=args.num_workers,
				task=task,
			)
			self.test_loader_dict_all[task] = test_loader
		# ========= Testing =========
		# if not train_task:
		if self.args.checkpoint != 'None':
			last_path = os.path.join(self.args.checkpoint)
			print(f"The last path is {last_path}")
			if os.path.exists(last_path + '.pth') and not self.args.now_train:
				self.load(last_path)
				task = '_'.join(os.path.basename(self.args.checkpoint).split('_')[:1])
				# self.test(task)
				self.test(task)
			else:
				print("No model found")
		else:
			task = self.task_list[-1]
			last_path = os.path.join(self.args.output, f'{task}_LAST.pth')
			if os.path.exists(last_path):
				self.load(last_path)
			self.test(task)

	def test_single(self, task, comp=False):
		self.test_loader = self.test_loader_dict_all[task]
		quesid2ans, quesid2gt = self.predict(self.test_loader, task)
		predict_gt_dict = {}
		if self.verbose:
			evaluator = self.test_loader.evaluator
			acc_dict_all = evaluator.evaluate_raw(quesid2ans)
			wandb_log_dict = {}
			wandb_log_dict['Test/overall'] = acc_dict_all['overall']
			print(task, wandb_log_dict)
		predict_gt_dict[task] = self.compile_preds(quesid2ans, quesid2gt)
		pred_dir = os.path.join(self.args.output, 'predictions')
		if not os.path.exists(pred_dir):
			os.makedirs(pred_dir, exist_ok=True)
		with open(f"{pred_dir}/{task}_gt_pred.json", 'w') as f:
			json.dump(predict_gt_dict, f, indent=4)


	def test(self, task, comp=False):
		# Test Set
		if not os.path.exists(self.args.output):
			os.mkdir(self.args.output)
		# =========== test for all previous tasks
		mega_log_dict = {}
		predict_gt_dict = {}
		mega_log_dict[task] = {}
		predict_gt_dict[task] = {}
		for test_task in self.sg_Ours:
			mega_log_dict[task][test_task] = []
			predict_gt_dict[task][test_task] = []
			self.test_loader = self.test_loader_dict_all[test_task]
			print(' ===== Test for the task "' + test_task + '"  ======')
			quesid2ans, quesid2gt = self.predict(self.test_loader, test_task)
			if self.verbose:
				evaluator = self.test_loader.evaluator
				acc_dict_all = evaluator.evaluate_raw(quesid2ans)
				wandb_log_dict = {}
				wandb_log_dict['Test/overall'] = acc_dict_all['overall']
				print(test_task, wandb_log_dict)
				mega_log_dict[task][test_task].append(wandb_log_dict)

				predict_gt_dict[task][test_task].append(self.compile_preds(quesid2ans, quesid2gt))
			self.result_matrix[task][test_task] = acc_dict_all['overall']
		pred_dir = os.path.join(self.args.output, 'predictions')
		if not os.path.exists(pred_dir):
			os.makedirs(pred_dir, exist_ok=True)
		with open(f"{pred_dir}/{task}_acc.json", 'w') as f:
			json.dump(mega_log_dict, f, indent=4)
		with open(f"{pred_dir}/{task}_gt_pred.json", 'w') as f:
			json.dump(predict_gt_dict, f, indent=4)

	def compile_preds(self, quesid2ans, quesid2gt):
		gt_pred_pairs = {}
		for key, val in quesid2ans.items():
			pred, _ = val
			img_id, question, gt = quesid2gt[key]
			gt_pred_pairs[key] = [img_id, question, pred, gt]
		return gt_pred_pairs

	def predict(self, loader, task, dump_path=None):
		self.model.eval()
		with torch.no_grad():
			quesid2ans, quesid2gt = {}, {}
			print("Predicting")
			if self.args.show_train_progress:
				pbar = tqdm(total=len(loader), ncols=120, desc="Prediction---")
			for i, batch in enumerate(loader):
				if self.args.distributed:
					results = self.model.module.test_step(batch, task)
				else:
					results = self.model.test_step(batch, task)
				pred_ans = results['pred_ans'] # generated_sents
				all_answers = batch['all_answers']
				ques_ids = batch['question_ids']
				for qid, ans, gt in zip(ques_ids, pred_ans, all_answers):
					quesid2ans[qid] = (ans, gt)
					quesid2gt[qid] = batch['img_id'][0], \
						batch['sent'][0], batch['answers'][0]
				if self.args.show_train_progress:
					pbar.update(1)
			if self.args.show_train_progress:
				pbar.close()
		qid2ans = quesid2ans
		if self.verbose:
			quesid2ans = {}
			for k, v in qid2ans.items():
				quesid2ans[k] = v
			if dump_path is not None:
				evaluator = loader.evaluator
				evaluator.dump_result(quesid2ans, dump_path)
		return quesid2ans, quesid2gt

	def evaluate(self, loader, task, dump_path=None):
		quesid2ans,_ = self.predict(loader, task, dump_path)
		if self.verbose:
			evaluator = loader.evaluator
			acc_dict = evaluator.evaluate_raw(quesid2ans)
			return acc_dict

def main_worker(args):
	args.gpu = 0
	sg_Ours = Sg_task[args.scenario][args.sequence]
	if args.train_multi:
		from src.multi.trainer_multi_sgvqa import SGTrainerMulti
		trainer = SGTrainerMulti(args, sg_Ours, train=True)
	else:
		trainer = Trainer(args, sg_Ours, train=True)
	if args.now_train:
		if args.checkpoint != 'None':
			trainer.train(load=True)
		else:
			trainer.train(load=False)

		print('#------------------ Training complete --------------------#')
		# show_results_matrix(trainer.result_matrix)
		# path = args.output + 'results_matrix.json'
		# # save_results_matrix(trainer.result_matrix, path)
		# metric_dict = evaluate_metric(trainer.result_matrix)
		# print('#------  Metric  ------#')
		# print('Incremental avg accuracy:', metric_dict['Incre_avg_acc'])
		# print('*** Avg accuracy ***', metric_dict['Avg_acc'])
		# print('Incremental avg forget:', metric_dict['Incre_avg_forget'])
		# print('*** Avg forget ***', metric_dict['Avg_forget'])
		# print('6Q Incremental avg accuracy:', metric_dict['Incre_avg_acc_6Q'])
		# print('*** _6Q Avg accuracy ***', metric_dict['Avg_acc_6Q'])
		# print('_6Q Incremental avg forget:', metric_dict['Incre_avg_forget_6Q'])
		# print('*** _6Q Avg forget ***', metric_dict['Avg_forget_6Q'])

	else:
		if args.checkpoint!='None':
			task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9))
			task = Sg_task[args.scenario][args.sequence][task_idx]
			args.checkpoint = f'{args.output}/{task}_BEST'
			print(args.checkpoint)
			trainer.Test(load=True)
		else:
			trainer.Test(load=False)

		try:
			print('#------------------ Final Performance --------------------#')
			print(trainer.result_matrix['q_causal'])
			acc = 0
			for key in trainer.result_matrix['q_causal']:
				acc += trainer.result_matrix['q_causal'][key]
			print('AP:', round(acc/10, 4))

		except:
			pass


if __name__ == "__main__":
	cudnn.benchmark = True
	args = parse_args()
	ngpus_per_node = 1
	args.world_size = ngpus_per_node
	args_dict = vars(args)
	if not os.path.exists(args.output):
		os.makedirs(args.output, exist_ok=True)
	args_output_path = os.path.join(args.output, 'object_BEST.pth')
	source_path = 'snap/naiveblip_sgvqa_no_ents/object_BEST.pth'
	if not os.path.exists(args_output_path):
		try:
			shutil.copyfile(source_path, args_output_path)
			print(f"Successfully copied {source_path} to {args_output_path}")
		except Exception as e:
			print(f"Failed to copy file: {e}")
	else:
		print(f"File already exists: {args_output_path}")
	with open(f'{args.output}/config.json', 'w') as f:
		json.dump(args_dict, f, indent=4)
	main_worker(args)



