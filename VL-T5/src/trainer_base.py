import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb
from pprint import pformat

from baselines.ewc import EWC
from baselines.mas import MAS

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class TerminationError(Exception):
    """Error raised when a termination signal is received."""

    def __init__(self):
        super().__init__("External signal received: forcing termination")

class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        # self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

    def create_config(self):
        from transformers import T5Config, BartConfig, Blip2Config, InstructBlipConfig 

        if 't5' in self.args.backbone:
            config_class = T5Config
        elif 'bart' in self.args.backbone:
            config_class = BartConfig
        elif 'blip' in self.args.backbone:
            config_class =  Blip2Config
        else:
            return None

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
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone
        if 'blip' in model_name:
            # model_name = "pretrained/models--Salesforce--blip2-opt-2.7b/snapshots/235c75ea3861136b9dd202c6edc6a7ba285c35e3"
            model_name = "Salesforce/blip2-opt-2.7b"
        print(f"Loading {model_class}")
        model = model_class.from_pretrained(model_name,
            config=config,
            device_map="auto",  
            trust_remote_code=True,
            **kwargs
        )
        if self.args.lambda_ewc > 0:
            print(f'Using EWC regularization with lambda {self.args.lambda_ewc}')
            ewc_module = EWC(ewc_lambda=self.args.lambda_ewc, decay_factor=0.99, uniform_importance=False)
            return model, ewc_module
        if self.args.lambda_mas > 0:
            print(f'Using MAS regularization with lambda {self.args.lambda_mas}')
            mas_module = MAS(lambda_reg=self.args.lambda_mas, alpha=0.99)
            return model, mas_module
        return model

    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast, AutoProcessor
        from tokenization import VLT5Tokenizer, VLT5TokenizerFast
        if 'blip' in self.args.tokenizer:
            processor = AutoProcessor.from_pretrained(
                self.args.backbone,
                max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case,
                **kwargs
                    )
            tokenizer = processor.tokenizer
        else:
            if 't5' in self.args.tokenizer:
                if self.args.use_vision:
                    # tokenizer_class = VLT5Tokenizer
                    tokenizer_class = VLT5TokenizerFast
                else:
                    # tokenizer_class = T5Tokenizer
                    tokenizer_class = T5TokenizerFast
            elif 'bart' in self.args.tokenizer:
                tokenizer_class = BartTokenizer
                # tokenizer_class = BartTokenizerFast
            tokenizer_name = self.args.backbone

            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case,
                **kwargs
                )

        return tokenizer

    def create_optimizer_and_scheduler(self, total_train_num):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None
        if 'blip_adamw' in self.args.optim:
            optim = torch.optim.AdamW(params=self.model.parameters(), lr=1e-4, 
                weight_decay=self.args.warmup_ratio)
            # nparam = count_parameters(self.model.parameters())
            # print(f'trainable_parameters = {nparam}')

        elif 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
            batch_per_epoch = int(total_train_num / self.args.batch_size) #len(train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]

            if not self.args.freeze:
                no_decay = ["bias", "LayerNorm.weight"] # ---- here
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                print("Freeze the Enc and Dec")
                for n, p in self.model.named_parameters():
                    p.requires_grad = False
                    # print(n)

                for n, p in self.model.module.shared.named_parameters():
                    p.requires_grad = True
                # self.model.module.encoder.prefix.requires_grad = True
                # for n, p in self.model.module.encoder.prompt_pool_module.named_parameters():
                #     p.requires_grad = True

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if
                                   not any(nd in n for nd in no_decay) and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if
                                   any(nd in n for nd in no_decay) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_constant_schedule_with_warmup(
                optim, warmup_iters)
            # lr_scheduler = get_linear_schedule_with_warmup(
            #     optim, warmup_iters, t_total)

        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)
        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self, seed=0, ifseed=False):
        if ifseed:
        # seed = 668
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print('random seed', seed)

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
        # if not self.args.distributed or self.args.local_rank == 0:
        savepath = os.path.join(self.args.output, "%s.pth" % name)
        state_dict_to_save = {"optimizer": self.optim.state_dict(), "examplar": self.Examplar_set}
        # Access model depending on whether it's distributed or not
        actual_model = self.model.module if self.args.distributed else self.model
        if self.args.blip_model != "naiveblip":
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
        if self.args.fp16:
            state_dict_to_save["scaler"] = self.scaler.state_dict()

        if 'blip' in self.args.backbone:
            if self.args.ft_layers == 'full':
                state_dict_to_save["model"] = {
                'query_tokens': actual_model.query_tokens.data, 
                'language_projection':actual_model.language_projection.state_dict(),
                'qformer':actual_model.qformer.state_dict()}
            elif self.args.ft_layers == 'query_tokens':
                state_dict_to_save["model"] = {
                'query_tokens': actual_model.query_tokens.data, 
                'language_projection_questions':actual_model.language_projection_questions.state_dict(),
                'language_projection_answers':actual_model.language_projection_answers.state_dict()}
                if hasattr(actual_model.vision_model, 'prompt'):
                    state_dict_to_save['model'].update(actual_model.vision_model.prompt.state_dict())
                
                if self.regularizer is not None:
                    regularizer_state_dict = self.regularizer.get_state_dict()
                    if regularizer_state_dict:
                        state_dict_to_save['regularizer'] = regularizer_state_dict
            
            elif self.args.ft_layers == 'last layer':
                num_layers = len(actual_model.qformer.encoder.layer)
                state_dict_to_save["model"] = {
                'query_tokens': actual_model.query_tokens.data,
                'language_projection':actual_model.language_projection.state_dict(),
                'last_layer': actual_model.qformer.encoder.layer[num_layers - 1].state_dict()}
        elif 't5' in self.args.backbone:
            state_dict_to_save["model"] = actual_model.state_dict()

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
        if 'blip' in self.args.backbone:
            # Load different components based on ft_layers
            if self.args.ft_layers == 'full':
                actual_model.qformer.load_state_dict(checkpoint["model"]["qformer"], strict=False)
                actual_model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
                actual_model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
            elif self.args.ft_layers == 'query_tokens':
                actual_model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
                # actual_model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
                if 'language_projection_answers' in checkpoint['model'] and 'language_projection_questions' in checkpoint['model']:
                    actual_model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection_answers'])
                    actual_model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])
                else:
                    print("Loading the original weights")
                    actual_model.language_projection_answers.load_state_dict(checkpoint['model']['language_projection'])
                if hasattr(actual_model.vision_model, 'prompt'):
                    print('Loading prompts weights')
                    prompt_dict = {'prompt': checkpoint['model']['prompt'], 'prompt_key': checkpoint['model']['prompt_key']}
                    actual_model.vision_model.prompt.load_state_dict(prompt_dict)
                
                if 'regularizer' in checkpoint:
                    self.regularizer.load_state_dict(checkpoint['regularizer'])

            if self.args.ft_layers =='last layer':
                num_layers = len(actual_model.qformer.encoder.layer)
                actual_model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
                actual_model.qformer.encoder.layer[num_layers - 1].load_state_dict(checkpoint['model']['last_layer'])
                actual_model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
            # self.optim.load_state_dict(checkpoint["optimizer"])
        elif 't5' in self.args.backbone:
            actual_model.load_state_dict(checkpoint["model"])
        if self.args.blip_model != "naiveblip":
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
            actual_model.Examplar_set = checkpoint["examplar"]
        if self.args.fp16:
            self.scaler.load_state_dict(checkpoint["scaler"])

        if self.verbose:
            print('Model loaded from ', path)



