import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'blip_adamw':
        if verbose:
            print("Optimizer: Using AdamW for BLIP")
        optimizer = 'blip_adamw'
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='mem_nohier_query.json')
    parser.add_argument('--ifseed', action='store_true')
    parser.add_argument('--seed', type=int, default=66666, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=50)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=80)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str)

    parser.add_argument('--log_train_accuracy', action='store_true')

    parser.add_argument('--n_ground', type=int, default=1)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',default=0.15, type=float)

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=20)

    # Data
    parser.add_argument('--caption_only', action='store_true')
    parser.add_argument('--coco_only', action='store_true')
    parser.add_argument('--caption_cocoonly', default=True, type=str2bool)

    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--oscar_tags', action='store_true')

    parser.add_argument('--prefix', type=str, default=None)

    # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool)
    parser.add_argument('--single_vqa_prefix', action='store_true')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true')

    # VQA
    parser.add_argument("--raw_label", action='store_true')
    parser.add_argument("--answer_normalize", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--test_answerable", action='store_true')

    # RefCOCOg
    parser.add_argument('--RefCOCO_GT', action='store_true')
    parser.add_argument('--RefCOCO_BUTD', action='store_true')
    parser.add_argument("--shuffle_boxes", action='store_true')
    parser.add_argument('--vis_pointer', type=str2bool, default=False)

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin')
    parser.add_argument("--tasks", type=str, default='')

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Memory.
    parser.add_argument("--memory", action='store_true')
    parser.add_argument("--m_size", type=int, default=1000)

    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--Q", type=str, default="All_Q_v4")
    parser.add_argument("--pull_constraint_coeff", type=float, default=1.0) # for Prompt

    parser.add_argument("--freeze", action='store_true')

    parser.add_argument("--lambda_Q", type=float, default=0.01)
    parser.add_argument("--lambda_V", type=float, default=0.1)
    parser.add_argument("--lambda_Q_new", type=float, default=0)
    parser.add_argument("--lambda_V_new", type=float, default=0)

    parser.add_argument("--comp_cate", type=str, default='G3')
    parser.add_argument("--ewc_loss_weight", type=float, default=100.0)
    parser.add_argument("--lambda_neighbor", type=float, default=10)
    parser.add_argument("--reg_lambda", type=float, default=10000)
    parser.add_argument("--now_train", action='store_true')


    parser.add_argument("--proto_alpha", type=float, default=0.5)
    parser.add_argument("--proto_beta", type=float, default=0.3)
    parser.add_argument("--eval_blip", default=False, type=str2bool)
    parser.add_argument("--train_from_scratch", default=False, type=str2bool)
    parser.add_argument("--use_class_hierarchy", default=True, type=str2bool)
    parser.add_argument("--show_train_progress", default=False, type=str2bool)
    parser.add_argument("--log_all_runs", default=False, type=str2bool)
    parser.add_argument("--ft_layers", type=str, choices=['full', 'query_tokens', 'query_tokens_random', 'query_tokens_task'], default='only query tokens')
    parser.add_argument("--train_multi", default=False, type=str2bool)
    parser.add_argument("--blip_model", type=str,default="naiveblip")
    parser.add_argument("--lambda_ewc", type=float, default=0.0)
    parser.add_argument("--lambda_mas", type=float, default=0.0)
    parser.add_argument("--lambda_l2p", type=float, default=0.0)
    parser.add_argument('--tasks_topk', type=int, default=-1)
    parser.add_argument('--prompt_pool', default=False, type=str2bool)
    parser.add_argument('--pool_size', default=None, type=int)
    parser.add_argument('--scenario', default='function', type=str)
    parser.add_argument('--sequence', default='oarlks', type=str)
    parser.add_argument('--avg_with_base', default=False, type=str2bool)
    parser.add_argument('--avg_with_last', default=False, type=str2bool)
    parser.add_argument('--use_gen_data', default=False, type=str2bool)
    parser.add_argument('--use_cap_loss', default=False, type=str2bool)
    parser.add_argument('--use_biased_data', default=False, type=str2bool)
    parser.add_argument('--method', default='no_ents', choices=['ents', 'no_ents', 'qtype', 'lamol', 'vqg'], type=str)
    parser.add_argument('--replay_strategy', default='static', choices=['static', 'dynamic'], type=str)
    parser.add_argument('--dynamic_sampling', default='random', choices=['random', 'balanced'], type=str)
    parser.add_argument('--balance_strategy', default='classifier', choices=['classifier', 'cluster', 'none'], type=str)
    parser.add_argument('--n_clusters', default=7, type=int)
    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
