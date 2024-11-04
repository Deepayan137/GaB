import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from transformers import AutoProcessor
from transformers import Blip2Config
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb
from pprint import pformat


proj_dir = Path(__file__).resolve().parent.parent

class TrainerBase(object):
	def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
		self.args = args
		self.set_seed(args.seed)
		self.verbose = True
		if self.args.tokenizer is None:
			self.args.tokenizer = self.args.backbone

		if not self.verbose:
			set_global_logging_level(logging.ERROR, ["transformers"])

	def set_seed(self, seed):
		print(f"The Random seed is {self.args.seed}")
		random.seed(seed)  # Python random module
		np.random.seed(seed)  # Numpy module
		torch.manual_seed(seed)  # PyTorch
		torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
		torch.backends.cudnn.deterministic = True  # enforce deterministic algorithm
		torch.backends.cudnn.benchmark = False  

	def create_config(self):
		config_class =  Blip2Config
		config = config_class.from_pretrained(self.args.backbone)
		args = self.args

		config.feat_dim = args.feat_dim
		config.pos_dim = args.pos_dim
		config.n_images = 2

		config.use_vis_order_embedding = args.use_vis_order_embedding

		config.dropout_rate = args.dropout
		config.dropout = args.dropout
		config.attention_dropout = args.dropout
		config.activation_dropout = args.dropout

		config.use_vis_layer_norm = args.use_vis_layer_norm
		config.individual_vis_layer_norm = args.individual_vis_layer_norm
		config.losses = args.losses

		config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
		config.classifier = args.classifier
		# config.ft_layers = "last"
		return config


	def create_model(self, model_class, config=None, **kwargs):
		model_name = self.args.backbone
		print(f"Loading {model_class}")
		model = model_class.from_pretrained(model_name,
			cache_dir='.',
			config=config,
			device_map="auto",  
			trust_remote_code=True,
			**kwargs
		)
		return model

	def create_tokenizer(self, **kwargs):
		processor = AutoProcessor.from_pretrained(
			self.args.backbone,
			max_length=self.args.max_text_length,
			do_lower_case=self.args.do_lower_case,
			**kwargs
				)
		tokenizer = processor.tokenizer
		return tokenizer

	def create_optimizer_and_scheduler(self, total_train_num):
		if self.verbose:
			print('Building Optimizer')
		lr_scheduler = None
		for name, param in self.model.named_parameters():
			if param.requires_grad:
				print(name)
		if 'blip_adamw' in self.args.optim:
			from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
			params = list(self.model.language_projection.parameters()) + list(self.model.language_projection_answers.parameters())
			optim = torch.optim.AdamW(params=params, lr=self.args.lr, 
				weight_decay=self.args.warmup_ratio)
		else:
			optim = self.args.optimizer(
				list(self.model.parameters()), self.args.lr)

		return optim, lr_scheduler


	def init_weights(self, seed=0, ifseed=False):
		# if ifseed:
		# # seed = 668
		# 	torch.manual_seed(seed)
		# 	torch.cuda.manual_seed_all(seed)
		# 	np.random.seed(seed)
		# 	print('random seed', seed)

		def init_bert_weights(module):
			""" Initialize the weights."""
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=1)
			elif isinstance(module, nn.LayerNorm):
				module.bias.data.zero_()
				module.weight.data.fill_(1.0)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		self.model.apply(init_bert_weights)
		self.model.init_weights()

	def predict(self):
		pass

	def evaluate(self):
		pass

	def save(self, name):
		if not os.path.isdir(self.args.output):
			os.makedirs(self.args.output, exist_ok=True)
		savepath = os.path.join(self.args.output, "%s.pth" % name)
		state_dict_to_save = {"optimizer": self.optim.state_dict(), "examplar": self.Examplar_set}
		# Access model depending on whether it's distributed or not
		actual_model = self.model.module if self.args.distributed else self.model
		if self.args.blip_model == "vqaclblip":
			try:
				state_dict_to_save["Q_prototype"] = actual_model.Q_prototype
				state_dict_to_save["V_prototype"] = actual_model.V_prototype
				state_dict_to_save["Q_task_mem_proto"] = actual_model.Q_task_mem_proto
				state_dict_to_save["V_task_mem_proto"] = actual_model.V_task_mem_proto
				state_dict_to_save["Q_prototype_num"] = actual_model.Q_prototype_num
				state_dict_to_save["V_prototype_num"] = actual_model.V_prototype_num
			except Exception as e:
				print(e)
				print('save prototype error')
			state_dict_to_save["model"] = {
			'query_tokens': actual_model.query_tokens.data,			
			'language_projection':actual_model.language_projection.state_dict()}
		else:
			state_dict_to_save["model"] = {
			'query_tokens': actual_model.query_tokens.data, 
			'language_projection_questions':actual_model.language_projection_questions.state_dict(),
			'language_projection_answers':actual_model.language_projection_answers.state_dict()}
		print(f"Saving model at {self.args.ft_layers} parameters @ {savepath}")
		torch.save(state_dict_to_save, savepath)



	def load(self, path, loc=None):
		if loc is None and hasattr(self.args, 'gpu'):
			loc = f'cuda:{self.args.gpu}'
		if not path.endswith('.pth'):
			path = "%s.pth" % path
		checkpoint = torch.load(path, map_location=loc)
		# Access model depending on whether it's distributed or not
		actual_model = self.model.module if self.args.distributed else self.model
		actual_model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
		if self.args.blip_model=='naiveblip':
			if 'language_projection_answers' in checkpoint['model'] and 'language_projection_questions' in checkpoint['model']:
				actual_model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection_answers'])
				actual_model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])
			else:
				print("Loading the original weights")
				actual_model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
		elif self.args == 'vqaclblip':
			actual_model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
			if "Q_prototype" in checkpoint.keys():
				actual_model.Q_prototype = checkpoint["Q_prototype"]
			if "V_prototype" in checkpoint.keys():
				actual_model.V_prototype = checkpoint["V_prototype"]
			if "Q_task_mem_proto" in checkpoint.keys():
				actual_model.Q_task_mem_proto = checkpoint["Q_task_mem_proto"]
			if "V_task_mem_proto" in checkpoint.keys():
				actual_model.V_task_mem_proto = checkpoint["V_task_mem_proto"]
			if "Q_prototype_num" in checkpoint.keys():
				actual_model.Q_prototype_num = checkpoint["Q_prototype_num"]
			if "V_prototype_num" in checkpoint.keys():
				actual_model.V_prototype_num = checkpoint["V_prototype_num"]
		if "examplar" in checkpoint.keys():
			print("Loading Examplar")
			actual_model.Examplar_set = checkpoint["examplar"]
		if self.verbose:
			print('Model loaded from ', path)



