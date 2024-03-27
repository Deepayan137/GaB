import sys
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args

from vqa_data_memory import get_loader, get_loader_test, VQADataset, get_loader_memory
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
import json
import random
import os

import signal

def __handle_signal(signum, frame):
    raise TerminationError()

signal.signal(signal.SIGINT, __handle_signal)
signal.signal(signal.SIGTERM, __handle_signal)

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

from trainer_base import TrainerBase, TerminationError
#  10 task
from Question_type import All_task, Comp_task, show_results_matrix, evaluate_metric, Category_splits, ImgId_cate_map, random_dic

from transformers import AutoProcessor
def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Trainer(TrainerBase):
    def __init__(self, args, coco_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.result_matrix = {}
        self.task_list = []
        tasks_set = coco_Ours
        if args.tasks_topk != -1:
            tasks_set = tasks_set[:args.tasks_topk]
        
        for task in tasks_set:
            self.result_matrix[task] = {}
            self.task_list.append(task)

        self.train_loader_dict = {}
        self.val_loader_dict = {}
        self.test_loader_dict = {}
        self.test_loader_dict_all = {}
        if 'blip' in args.backbone:
            from vqa_data_blip import VQADataset
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.train_dset = VQADataset(args.train, True)
        self.val_dset = VQADataset(args.valid, True)
        self.test_dset = VQADataset(args.test, True)

        super().__init__(
            args,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from vqa_model import VLT5VQA
        from vqa_model_blip import BLIP2VQA, NaiveBLIP2
        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5VQA
        elif 'blip' in args.backbone:
            from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
            # model_class = BLIP2VQA
            model_class = NaiveBLIP2

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        self.regularizer = None 
        if isinstance(self.model, tuple):
            self.regularizer = self.model[1]
            self.model = self.model[0]

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)
        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)
        if self.regularizer is not None:
            self.regularizer = self.regularizer.to(args.gpu)
        self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
        self.iftrain = train
        self.coco_Ours = coco_Ours

        self.task_iftrain = {}
        for task in self.coco_Ours:
            self.task_iftrain[task] = 0

        self.task_total_num = torch.zeros(len(self.task_list))

        self.M = args.m_size
        if args.use_class_hierarchy:
            self.Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
        else:
            self.Examplar_set = []

        self.composition_test_cate = args.comp_cate

    def _load_checkpoint(self, checkpoint_name, latest_task_idx):
        checkpoint_model = f'{self.args.output}/{checkpoint_name}_LAST'
        for idx, task in enumerate(self.task_list):
            if idx <= latest_task_idx:
                self.task_iftrain[task] = 1
        self.load(checkpoint_model)
        print(f'Success to load the checkpoint from the task {checkpoint_name}')

    def train(self, load=False):
        
        if 'blip' in args.backbone:
            from vqa_data_blip import get_loader, get_loader_test, VQADataset, get_loader_memory
        latest_task_idx = -1
        if load:
            latest_task = '_'.join(os.path.basename(self.args.checkpoint).split('_')[:2])
            latest_task_idx = self.task_list.index(latest_task)
            self._load_checkpoint(latest_task, latest_task_idx)
        
        task2id = {self.task_list[i]:i for i in range(len(self.task_list))}
        id2task = {v:k for k, v in task2id.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.train_from_scratch:
            base_state = {
                "model":self.model.state_dict()
            }
            base_state_dict = {k: v.cpu() for k, v in base_state["model"].items()}
        try:
            for i, task in enumerate(self.task_list[latest_task_idx+1:]):  # for each task, train for several epochs
                if i > 0 and args.train_from_scratch:
                    # print("ending")
                    # sys.exit()
                    self.model.load_state_dict({k: v.to(device) for k, v in base_state_dict.items()})
                    self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
                    print("Reset model to base state and starting training for new task")
                task_idx = task2id[task]
                print('======================== Now is task "', task, '" ========================')
                self.task_iftrain[task] = 1
                # Memory
                if args.memory:
                    if task_idx >0:
                        each_memory = int(self.M / task_idx)

                        data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{self.task_list[task_idx - 1]}.json')
                        with open(data_info_path) as f:
                            data_info_dicts = json.load(f)

                        random.shuffle(data_info_dicts)  # shuffle
                        if args.use_class_hierarchy:
                            each_memory_for_cate = int(each_memory / len(Category_splits))
                            for cate in Category_splits:
                                num = 0
                                self.Examplar_set[cate].append([])
                                for _d in data_info_dicts:
                                    img_id = _d['img_id']
                                    if img_id in ImgId_cate_map:
                                        if ImgId_cate_map[img_id] in Category_splits[cate]:
                                            self.Examplar_set[cate][task_idx - 1].append(_d)
                                            num += 1
                                            if num >= each_memory_for_cate:
                                                break

                            print('Load from Partition_Q_v3......')
                            for cate in Category_splits:
                                for i in range(task_idx):
                                    self.Examplar_set[cate][i] = self.Examplar_set[cate][i][: each_memory_for_cate]

                            All_examplar = []
                            for E_set in self.Examplar_set:
                                for task_set in self.Examplar_set[E_set]:
                                    All_examplar += task_set
                        else:
                            All_examplar = data_info_dicts[:each_memory]
                        # assert len(All_examplar) == M
                        print("# The size of the cate Memory:", len(All_examplar))
                    else:
                        All_examplar = []
                        each_memory = 0
                else:
                    All_examplar = []
                    each_memory = 0

                # Load the data
                print("#Loading ", task)

                train_loader, total_num_Q = get_loader(
                    args,
                    self.coco_Ours,
                    [],
                    self.train_dset,
                    split=args.train, mode='train', batch_size=args.batch_size,
                    distributed=args.distributed, gpu=args.gpu,
                    workers=args.num_workers,
                    topk=args.train_topk,
                    task=task,
                )

                self.task_total_num[task_idx] = total_num_Q


                if args.valid_batch_size is not None:
                    self.valid_batch_size = args.valid_batch_size
                else:
                    self.valid_batch_size = args.batch_size
                print(f'Building val loader at GPU {args.gpu}')
                val_loader, _ = get_loader(
                    args,
                    self.coco_Ours,
                    [],
                    self.val_dset,
                    split=args.valid, mode='val', batch_size=self.valid_batch_size,
                    distributed=args.distributed, gpu=args.gpu,
                    workers=4,
                    topk=args.valid_topk,
                    task=task,
                )

                print(f'Building test loader at GPU {args.gpu}')
                test_loader = get_loader_test(
                    args,
                    self.coco_Ours,
                    [],
                    self.test_dset,
                    split=args.test, mode='val', batch_size=self.valid_batch_size,
                    distributed=args.distributed, gpu=args.gpu,
                    workers=4,
                    topk=args.valid_topk,
                    task=task,
                )
                self.test_loader_dict_all[task] = test_loader

                print("#Loading ", task)
                memory_loader = get_loader_memory(
                    args,
                    self.coco_Ours,
                    All_examplar,
                    self.train_dset,
                    split=args.train, mode='train', batch_size=32,
                    distributed=args.distributed, gpu=args.gpu,
                    workers=args.num_workers,
                    topk=args.train_topk,
                )  #G1-G5

                if self.verbose:
                    loss_meter = LossMeter()
                    loss_meter_mem = LossMeter()
                    best_valid = 0.
                    best_epoch = 0

                    if 't5' in self.args.backbone:
                        if self.args.use_vision:
                            project_name = "VLT5_VQA"
                        else:
                            project_name = "T5_VQA"
                    elif 'blip' in self.args.backbone:
                        if self.args.use_vision:
                            project_name = "BLIP_VQA"
                        
                    elif 'bart' in self.args.backbone:
                        if self.args.use_vision:
                            project_name = "VLBart_VQA"
                        else:
                            project_name = "Bart_VQA"

                    src_dir = Path(__file__).resolve().parent
                    base_path = str(src_dir.parent)
                    src_dir = str(src_dir)
                    # wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

                if self.args.distributed:
                    dist.barrier()

                global_step = 0
                
                if args.use_class_hierarchy:
                    Category_splits_random = random_dic(Category_splits)
                else:
                    Category_splits_random = {'G1': list(np.arange(80))}
                for idx, cateGroup in enumerate(Category_splits_random):
                    print('-------- Training the cate group ', cateGroup,' of task ', task,'------')
                    self.train_loader_cate = train_loader[cateGroup]
                    self.val_loader_cate = val_loader[cateGroup]
                    self.memory_loader_cate = memory_loader[cateGroup]
                    # Optimizer
                    if self.iftrain:
                        if len(self.memory_loader_cate.dataset) > 0:
                            total_train_num = 2 * len(self.train_loader_cate.dataset)
                        else:
                            total_train_num = len(self.train_loader_cate.dataset)
                        if self.args.fp16 and _use_native_amp:
                            self.scaler = torch.cuda.amp.GradScaler()
                        elif _use_apex:
                            self.model, self.optim = amp.initialize(
                                self.model, self.optim, opt_level='O1', verbosity=self.verbose)
                        else:
                            self.scaler = None
                    if cateGroup == self.composition_test_cate and task != self.task_list[latest_task_idx+1]:
                        print("-------- Pass the training for", cateGroup, 'for after composition testing.--------')
                        continue
                    start_epoch = 0
                    for epoch in range(start_epoch, self.args.epochs):
                        if self.start_epoch is not None:
                            epoch += self.start_epoch
                        self.model.train()

                        if self.args.distributed:
                            self.train_loader_cate.sampler.set_epoch(epoch)
                        if args.show_train_progress:
                            pbar = tqdm(total=len(self.train_loader_cate), ncols=120)
                        epoch_results = {
                            'loss': 0.,
                        }

                        quesid2ans = {}
                        if len(self.memory_loader_cate.dataset) > 0:
                            now_loader = zip(self.train_loader_cate, cycle(self.memory_loader_cate))
                            print('Use memory loader')
                        else:
                            now_loader = self.train_loader_cate

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
                            desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                            desc_str += f' | Loss {loss_meter.val:4f}'
                            if mem_batch:
                                loss_meter_mem.update(results_mem['loss'].item())
                                desc_str += f' | Loss_mem {loss_meter_mem.val:4f}'
                            else:
                                loss_meter_mem.update(-1)

                            if args.show_train_progress:
                                pbar.set_description(desc_str)
                                pbar.update(1)

                            if self.args.distributed:
                                dist.barrier()

                        if args.show_train_progress:
                            pbar.close()
                        print(f"Epoch {epoch}| Loss: {loss_meter.val}, Loss_mem: {loss_meter_mem.val}")
                        score_dict = self.evaluate(self.val_loader_cate, task)
                        print(score_dict)
                        
                        valid_score = score_dict['topk_score'] * 100.
                        valid_score_raw = score_dict['overall']

                        log_str = ''
                        log_str += "\nGroup %s Epoch %d: Valid Raw %0.2f Topk %0.2f" % (cateGroup, epoch, valid_score_raw, valid_score)

                        print(log_str)

                        if self.args.distributed:
                            dist.barrier()
                    # end of task: compute relevance of weights
                    if self.regularizer is not None:
                        self.regularizer.after_training_exp(model=self.model,
                                                            optimizer=self.optim, 
                                                            dloader=self.train_loader_cate,
                                                            current_task_id=task_idx,
                                                            proto_alpha=self.args.proto_alpha,
                                                            proto_beta= self.args.proto_beta,
                                                            mem_num_Q = 0,
                                                            total_num_Q=self.task_total_num
                                                            ) 
                self.save(task + "_LAST")
                prev_task_id = task_idx - 1
                if prev_task_id >= 0:
                    prev_task = id2task[prev_task_id]
                    print("Removing checkpoint for a previous task")
                    if os.path.exists(os.path.join(self.args.output, f"{prev_task}_LAST.pth")):
                        os.remove(os.path.join(self.args.output, f"{prev_task}_LAST.pth"))
                # print("========= Testing =========")
                self.Test(train_task=task)
        except TerminationError:
            print("Termination signal received.")
        # try:
        #     Q_prototype = self.model.module.Q_prototype
        #     V_prototype = self.model.module.V_prototype
        #     torch.save(Q_prototype, args.output + "/Q_prototype.pt")
        #     torch.save(V_prototype, args.output + "/V_prototype.pt")
        #     print(" ======= Saved the learned prototypes ======= ")
        # except:
        #     print('save prototype error')


    def train_step(self, batch, epoch_results, task_idx, each_memory):
        self.optim.zero_grad(set_to_none=True)
        embeddings=None
        if self.args.lambda_l2p:
            # prompt the embeddings
            out = self.regularizer.before_backward(self.model, batch['pixel_values'])
            prompt = out['prompt']
            cl_reg = out['loss']
            embeddings=prompt
        if self.args.fp16 and _use_native_amp:
            with autocast():
                if self.args.distributed:
                    results = self.model.module.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num, embeddings=embeddings)
                else:
                    results = self.model.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num, embeddings=embeddings)
        else:  # this
            if self.args.distributed:
                results = self.model.module.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num, embeddings=embeddings)
            else:
                results = self.model.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num, embeddings=embeddings)

        loss = results['loss']
        
        # add the regularization term accounting for weight relevance
        if self.regularizer is not None:
            if self.args.lambda_l2p == 0:
                cl_reg = self.regularizer.before_backward(self.model, device=loss.device)
            loss += cl_reg
        lambda_Q = self.args.lambda_Q
        lambda_V = self.args.lambda_V
        lambda_Q_new = self.args.lambda_Q_new
        lambda_V_new = self.args.lambda_V_new

        if 'loss_memory' in results:
            (loss_memory_Q, loss_memory_V) = results['loss_memory']
            loss = loss + lambda_Q * loss_memory_Q + lambda_V * loss_memory_V
        if 'loss_memory_new' in results:
            (loss_memory_Q_new, loss_memory_V_new) = results['loss_memory_new']
            loss = loss + lambda_Q_new * loss_memory_Q_new + lambda_V_new * loss_memory_V_new

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:  # this
            loss.backward()

        loss = loss.detach()

        # Update Parameters
        if self.args.clip_grad_norm > 0:
            if self.args.fp16 and _use_native_amp:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)
            elif self.args.fp16 and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(
                    self.optim), self.args.clip_grad_norm)
            else:  # this
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)

        if self.args.fp16 and _use_native_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # this
            self.optim.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()
        for param in self.model.parameters():
            param.grad = None

        # ----------
        # self.model.module.Q_prototype_param.data.clamp_(0, 1)
        # self.model.module.V_prototype_param.data.clamp_(0, 1)

        # global_step += 1

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

    def Test(self, load=False, train_task=None):

        for task_idx, task in enumerate(self.task_list):
            print('======================== Now is task "', task, '" ========================')
            if 'blip' in self.args.backbone:
                from vqa_data_blip import get_loader_test
            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=args.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

        # ========= Testing =========
        if not train_task:
            if self.args.checkpoint != 'None':
                last_path = os.path.join(self.args.checkpoint)
                if os.path.exists(last_path+'.pth') and not self.args.now_train:
                    self.load(last_path)
                    task = '_'.join(os.path.basename(self.args.checkpoint).split('_')[:2])
                    self.test_single(task)
            else:
                task = self.task_list[-1]
                last_path = os.path.join(self.args.output, f'{task}_LAST')
                self.load(last_path)
                self.test(task)
        else:
            self.test(train_task)

    def test_single(self, task, comp=False):
        self.test_loader = self.test_loader_dict_all[task]
        quesid2ans = self.predict(self.test_loader, task)
        if self.verbose:
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(quesid2ans)

            acc_dict_all = evaluator.evaluate_raw(quesid2ans)
            acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
            acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

            wandb_log_dict = {}
            wandb_log_dict['Test/overall'] = acc_dict_all['overall']
            wandb_log_dict['Test/topk_optimal'] = acc_dict_answerable['overall']
            wandb_log_dict['Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

            for qtype, score in acc_dict_all['perQuestionType'].items():
                wandb_log_dict[f'Test_Qtypes/{qtype}'] = score
            for atype, score in acc_dict_all['perAnswerType'].items():
                if atype == 'yes/no':
                    atype = 'yes_no'
                wandb_log_dict[f'Test_Atypes/{atype}'] = score

            print(task, wandb_log_dict)
    
    def test(self, task, comp=False):
        # Test Set
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        

        if not self.args.now_train and not self.args.eval_blip:
            self.model.module.Q_prototype = torch.load(self.args.output+'/Q_prototype.pt')
            self.model.module.V_prototype = torch.load(self.args.output+'/V_prototype.pt')


        # =========== test for all previous tasks
        flag = 1
        mega_log_dict = {}
        mega_log_dict[task] = {}
        for test_task in self.task_list:
            mega_log_dict[task][test_task] = []
            # if self.args.now_train:
            #     if self.task_iftrain[test_task] == 0:
            #         flag = 0
            if flag == 1:
                self.test_loader = self.test_loader_dict_all[test_task]
                print(' ===== Test for the task "' + test_task + '"  ======')

                quesid2ans = self.predict(self.test_loader, test_task)

                if self.verbose:
                    evaluator = self.test_loader.evaluator
                    score_dict = evaluator.evaluate(quesid2ans)

                    acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                    acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                    acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                    wandb_log_dict = {}
                    wandb_log_dict['Test/overall'] = acc_dict_all['overall']
                    wandb_log_dict['Test/topk_optimal'] = acc_dict_answerable['overall']
                    wandb_log_dict['Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

                    for qtype, score in acc_dict_all['perQuestionType'].items():
                        wandb_log_dict[f'Test_Qtypes/{qtype}'] = score
                    for atype, score in acc_dict_all['perAnswerType'].items():
                        if atype == 'yes/no':
                            atype = 'yes_no'
                        wandb_log_dict[f'Test_Atypes/{atype}'] = score

                    print(test_task, wandb_log_dict)
                    mega_log_dict[task][test_task].append(wandb_log_dict)
                self.result_matrix[task][test_task] = acc_dict_all['overall']

                if self.args.distributed:
                    dist.barrier()
        if self.args.log_all_runs:
            with open(f"{self.args.output}/{self.args.exp_name}.json", 'a') as f:
                json.dump(mega_log_dict, f, indent=4)



    def predict(self, loader, task, dump_path=None):
        self.model.eval()
        ans_list = []
        with torch.no_grad():
            quesid2ans = {}
            print("Predicting")
            pbar = tqdm(total=len(loader), ncols=120, desc="Prediction---")
            # import pdb;pdb.set_trace()
            for i, batch in enumerate(loader):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch, task)

                if 'blip' in self.args.backbone:
                    pred_ans = self.processor.tokenizer.batch_decode(results['token_ids'], skip_special_tokens=True)
                else:
                    pred_ans = results['pred_ans'] # generated_sents
                
                ans_list.append(pred_ans)
                ques_ids = batch['question_ids']
                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans 
                pbar.update(1)
            pbar.close()
        print(ans_list[:10])
        if self.args.distributed:
            dist.barrier()

        qid2ans_list = dist_utils.all_gather(quesid2ans)
        if self.verbose:
            quesid2ans = {}
            for qid2ans in qid2ans_list:
                for k, v in qid2ans.items():
                    quesid2ans[k] = v

            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)
        return quesid2ans

    def evaluate(self, loader, task, dump_path=None):
        quesid2ans = self.predict(loader, task, dump_path)

        if self.verbose:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)
            topk_score = evaluator.evaluate(quesid2ans)
            acc_dict['topk_score'] = topk_score

            return acc_dict

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')

    coco_Ours = All_task


    trainer = Trainer(args, coco_Ours, train=True)

    if args.now_train:
        if args.checkpoint != 'None':
            trainer.train(load=True)
        else:
            trainer.train(load=False)

        print('#------------------ result_matrix --------------------#')
        show_results_matrix(trainer.result_matrix)
        path = args.output + 'results_matrix.json'
        # save_results_matrix(trainer.result_matrix, path)
        metric_dict = evaluate_metric(trainer.result_matrix)
        print('#------  Metric  ------#')
        print('Incremental avg accuracy:', metric_dict['Incre_avg_acc'])
        print('*** Avg accuracy ***', metric_dict['Avg_acc'])
        print('Incremental avg forget:', metric_dict['Incre_avg_forget'])
        print('*** Avg forget ***', metric_dict['Avg_forget'])
        print('6Q Incremental avg accuracy:', metric_dict['Incre_avg_acc_6Q'])
        print('*** _6Q Avg accuracy ***', metric_dict['Avg_acc_6Q'])
        print('_6Q Incremental avg forget:', metric_dict['Incre_avg_forget_6Q'])
        print('*** _6Q Avg forget ***', metric_dict['Avg_forget_6Q'])

    else:
        if args.checkpoint!='None':
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
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        # if args.load is not None:
        #     ckpt_str = "_".join(args.load.split('/')[-3:])
        #     comments.append(ckpt_str)

        # else:
        #     ckpt_str = 'scrach'
        #     comments.append(ckpt_str)
        # if args.comment != '':
        #     comments.append(args.comment)
        # comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    # if args.distributed:
    main_worker(args.local_rank, args)
