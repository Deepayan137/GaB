import torch.backends.cudnn as cudnn

# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import os
from transformers import AutoProcessor
from pathlib import Path
from packaging import version
import numpy as np
from tqdm import tqdm
import torch
import logging
import shutil

from param import parse_args

from vqa_data_blip import get_loader, get_loader_test, VQADataset, get_loader_memory
from utils import LossMeter, set_global_logging_level

# import dist_utils
from data_utils import get_memory_data
import json


os.environ["WANDB_MODE"] = "offline"
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
proj_dir = Path(__file__).resolve().parent.parent
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        pass
    _use_apex = True
else:
    _use_native_amp = True

# from trainer_base import TrainerBase, TerminationError
from Question_type import All_task, show_results_matrix, evaluate_metric, Category_splits, random_dic

# from vqa_model import VLT5VQA
from vqa_model_blip import NaiveBLIP2, BLIP2Prototype
from trainer_base import TrainerBase


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class Trainer(TrainerBase):
    def __init__(self, args, coco_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.result_matrix = {}
        self.task_list = []
        for task in coco_Ours:
            self.result_matrix[task] = {}
            self.task_list.append(task)

        self.train_loader_dict = {}
        self.val_loader_dict = {}
        self.test_loader_dict = {}
        self.test_loader_dict_all = {}
        self.train_dset = VQADataset(args.train, True)
        self.val_dset = VQADataset(args.valid, True)
        self.test_dset = VQADataset(args.test, True)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        super().__init__(args, train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        model_kwargs = {}
        if args.blip_model == "naiveblip":
            model_class = NaiveBLIP2
        elif args.blip_model == "vqaclblip":
            model_class = BLIP2Prototype

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)
        self.regularizer = None
        if isinstance(self.model, tuple):
            self.model, self.regularizer = self.model
        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        # GPU Options
        # print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time

            start = time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)
        if self.verbose:
            print(f"It took {time() - start:.1f}s")
        self.iftrain = train
        self.coco_Ours = coco_Ours

        self.task_iftrain = {}
        for task in self.coco_Ours:
            self.task_iftrain[task] = 0

        self.task_total_num = torch.zeros(len(self.task_list))
        self.M = args.m_size
        if args.use_class_hierarchy:
            self.Examplar_set = {"G1": [], "G2": [], "G3": [], "G4": [], "G5": []}
        else:
            self.Examplar_set = {}

        self.composition_test_cate = args.comp_cate

    def _load_checkpoint(self, checkpoint_name, latest_task_idx):
        checkpoint_model = f"{self.args.output}/{checkpoint_name}_LAST"
        for idx, task in enumerate(self.task_list):
            if idx <= latest_task_idx:
                self.task_iftrain[task] = 1
        self.load(checkpoint_model)
        self.Examplar_set = self.model.Examplar_set
        print(f"Success to load the checkpoint from the task {checkpoint_name}")

    def train(self, load=False):
        latest_task_idx = -1
        if load:
            latest_task = "_".join(os.path.basename(self.args.checkpoint).split("_")[:2])
            latest_task_idx = self.task_list.index(latest_task)
            self._load_checkpoint(latest_task, latest_task_idx)
        # run = wandb.init(
        #     # Set the project where this run will be logged
        #     project="vqacl",
        #     # Track hyperparameters and run metadata
        #     config=vars(args)
        # )
        task2id = {self.task_list[i]: i for i in range(len(self.task_list))}
        id2task = {v: k for k, v in task2id.items()}
        if args.train_from_scratch:
            base_state = {"model": self.model.state_dict()}
            base_state_dict = {k: v.cpu() for k, v in base_state["model"].items()}
        for i, task in enumerate(self.task_list[latest_task_idx + 1 :]):  # for each task, train for several epochs
            if i > 0 and args.train_from_scratch:
                self.model.load_state_dict({k: v.to(self.device) for k, v in base_state_dict.items()})
                self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
                print("Reset model to base state and starting training for new task")
            task_idx = task2id[task]
            print('======================== Now is task "', task, '" ========================')
            self.task_iftrain[task] = 1
            # Memory
            if args.memory:
                if task_idx > 0:
                    each_memory = int(self.M / task_idx)
                    All_examplar, self.Examplar_set = get_memory_data(args, task_idx, each_memory, self.Examplar_set, self.model, self.processor)
                    print("# The size of the cate Memory:", len(All_examplar))
                    if args.use_gen_data:
                        print("The answers from captioning model will be used to train the model")
                else:
                    All_examplar = []
                    each_memory = 0
            else:
                All_examplar = []
                each_memory = 0
            # Load the data
            print("#Loading ", task)
            print("Building train loader")
            train_loader, total_num_Q = get_loader(
                args,
                self.coco_Ours,
                [],
                self.train_dset,
                split=args.train,
                mode="train",
                batch_size=args.batch_size,
                workers=args.num_workers,
                topk=args.train_topk,
                task=task,
            )

            self.task_total_num[task_idx] = total_num_Q
            print("Building val loader")
            val_loader, _ = get_loader(
                args,
                self.coco_Ours,
                [],
                self.val_dset,
                split=args.valid,
                mode="val",
                batch_size=1,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )

            print("Building test loader")
            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test,
                mode="val",
                batch_size=1,
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
                split=args.train,
                mode="train",
                batch_size=30,
                workers=args.num_workers,
                topk=args.train_topk,
            )  # G1-G5

            # if self.verbose:
            loss_meter = LossMeter()
            loss_meter_mem = LossMeter()
            loss_meter_ques = LossMeter()
            best_epoch = 0
            global_step = 0

            if args.use_class_hierarchy:
                Category_splits_random = random_dic(Category_splits)
            else:
                Category_splits_random = {"G1": list(np.arange(80))}
            for idx, cateGroup in enumerate(Category_splits_random):
                print("-------- Training the cate group ", cateGroup, " of task ", task, "------")
                self.train_loader_cate = train_loader[cateGroup]
                self.val_loader_cate = val_loader[cateGroup]
                self.memory_loader_cate = memory_loader[cateGroup]
                # Optimizer
                if self.iftrain:
                    if len(self.memory_loader_cate.dataset) > 0:
                        total_train_num = 2 * len(self.train_loader_cate.dataset)
                    else:
                        total_train_num = len(self.train_loader_cate.dataset)

                    self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
                    if self.args.use_cap_loss:
                        print("We will use caption loss and two optimizers")
                        self.optim_question = torch.optim.AdamW(
                            params=self.model.language_projection_questions.parameters(),
                            lr=1e-4,
                            weight_decay=self.args.warmup_ratio,
                        )  # Using same weight decay as an example

                for epoch in range(self.args.epochs):
                    self.model.train()
                    if args.show_train_progress:
                        pbar = tqdm(total=len(self.train_loader_cate), ncols=120)
                    epoch_results = {"loss": 0.0}
                    quesid2ans = {}

                    if len(self.memory_loader_cate.dataset) > 0:
                        now_loader = zip(self.train_loader_cate, cycle(self.memory_loader_cate))
                        print("Use memory loader")
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

                        loss_meter.update(results["loss"].item())
                        if "loss_cap" in results:
                            loss_meter_ques.update(results["loss_cap"].item())
                        else:
                            loss_meter_ques.update(-1)
                        desc_str = f"Epoch {epoch} | LR {lr:.6f} | Loss {loss_meter.val:.4f} | Loss Ques {loss_meter_ques.val:.4f}"
                        if mem_batch:
                            loss_meter_mem.update(results_mem["loss"].item())
                            desc_str += f" | Loss_mem {loss_meter_mem.val:4f}"
                        else:
                            loss_meter_mem.update(-1)

                        if args.show_train_progress:
                            pbar.set_description(desc_str)
                            pbar.update(1)

                    if args.show_train_progress:
                        pbar.close()

                    print(f"Epoch {epoch}| Loss: {loss_meter.val}, Loss_mem: {loss_meter_mem.val}, Loss_Ques: {loss_meter_ques.val}")
                    score_dict = self.evaluate(self.val_loader_cate, task)
                    valid_score = score_dict["topk_score"] * 100.0
                    valid_score_raw = score_dict["overall"]
                    log_str = ""
                    log_str += "\nGroup %s Epoch %d: Valid Raw %0.2f Topk %0.2f" % (
                        cateGroup,
                        epoch,
                        valid_score_raw,
                        valid_score,
                    )
                    print(log_str)
                    # wandb.log({f"val_accuracy_{task}": valid_score_raw,
                    #         f"train_loss_{task}": loss_meter.val, f"train_loss_question_{task}":loss_meter_ques.val})
            print("Saving Last")
            self.save(task + "_LAST")

    def train_step(self, batch, epoch_results, task_idx, each_memory):
        self.optim.zero_grad(set_to_none=True)
        # self.optim_question.zero_grad(set_to_none=True)
        if self.args.blip_model == "vqaclblip":
            results = self.model.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num)
            lambda_Q = self.args.lambda_Q
            lambda_V = self.args.lambda_V
            lambda_Q_new = self.args.lambda_Q_new
            lambda_V_new = self.args.lambda_V_new
        else:
            results = self.model.train_step(batch, task_idx)
        # , self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num
        loss = results["loss"]
        if "loss_memory" in results:
            (loss_memory_Q, loss_memory_V) = results["loss_memory"]
            loss = loss + lambda_Q * loss_memory_Q + lambda_V * loss_memory_V
        if "loss_memory_new" in results:
            (loss_memory_Q_new, loss_memory_V_new) = results["loss_memory_new"]
            loss = loss + lambda_Q_new * loss_memory_Q_new + lambda_V_new * loss_memory_V_new
        if "loss_cap" in results:
            if self.args.use_cap_loss:
                loss_cap = results["loss_cap"]
                self.optim_question.zero_grad(set_to_none=True)
                loss_cap.backward(retain_graph=True)
                self.optim_question.step()
                loss_cap.detach()
        loss.backward()
        loss = loss.detach()
        # Update Parameters
        if self.args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        self.optim.step()
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

    def Test(self, load=False, train_task=None):
        for task_idx, task in enumerate(self.task_list):
            print('======================== Now is task "', task, '" ========================')
            if "blip" in self.args.backbone:
                from vqa_data_blip import get_loader_test
            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test,
                mode="val",
                batch_size=args.valid_batch_size,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

        # ========= Testing =========
        if not train_task:
            if self.args.checkpoint != "None":
                last_path = os.path.join(self.args.checkpoint)
                if os.path.exists(last_path + ".pth") and not self.args.now_train:
                    self.load(last_path)
                    task = "_".join(os.path.basename(self.args.checkpoint).split("_")[:2])
                    # task = self.task_list[1]
                self.test(task)
            else:
                task = self.task_list[-1]
                last_path = os.path.join(self.args.output, f"{task}_LAST")
                self.test_single(task)
        else:
            self.test(train_task)

    def test_single(self, task, comp=False):
        self.test_loader = self.test_loader_dict_all[task]
        quesid2ans, quesid2gt = self.predict(self.test_loader, task)
        # predict_gt_dict = {}
        if self.verbose:
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(quesid2ans)

            acc_dict_all = evaluator.evaluate_raw(quesid2ans)
            acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
            acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

            wandb_log_dict = {}
            wandb_log_dict["Test/overall"] = acc_dict_all["overall"]
            wandb_log_dict["Test/topk_optimal"] = acc_dict_answerable["overall"]
            wandb_log_dict["Test/topk_not_optimal"] = acc_dict_unanswerable["overall"]

            for qtype, score in acc_dict_all["perQuestionType"].items():
                wandb_log_dict[f"Test_Qtypes/{qtype}"] = score
            for atype, score in acc_dict_all["perAnswerType"].items():
                if atype == "yes/no":
                    atype = "yes_no"
                wandb_log_dict[f"Test_Atypes/{atype}"] = score

            print(task, wandb_log_dict)
        # predict_gt_dict[task] = self.compile_preds(quesid2ans, quesid2gt)
        # pred_dir = os.path.join(self.args.output, 'predictions')
        # if not os.path.exists(pred_dir):
        #     os.makedirs(pred_dir, exist_ok=True)
        # # fname = os.path.basename(self.args.backbone)
        # with open(f"{pred_dir}/{task}_gt_pred.json", 'w') as f:
        #     json.dump(predict_gt_dict, f, indent=4)

    def test(self, task, comp=False):
        # Test Set
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        # =========== test for all previous tasks
        flag = 1
        mega_log_dict = {}
        # predict_gt_dict = {}
        mega_log_dict[task] = {}
        # predict_gt_dict[task] = {}
        for test_task in self.coco_Ours:
            mega_log_dict[task][test_task] = []
            # predict_gt_dict[task][test_task] = []
            if flag == 1:
                self.test_loader = self.test_loader_dict_all[test_task]
                print(' ===== Test for the task "' + test_task + '"  ======')

                quesid2ans, quesid2gt = self.predict(self.test_loader, test_task)

                if self.verbose:
                    evaluator = self.test_loader.evaluator
                    score_dict = evaluator.evaluate(quesid2ans)

                    acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                    acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                    acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                    wandb_log_dict = {}
                    wandb_log_dict["Test/overall"] = acc_dict_all["overall"]
                    wandb_log_dict["Test/topk_optimal"] = acc_dict_answerable["overall"]
                    wandb_log_dict["Test/topk_not_optimal"] = acc_dict_unanswerable["overall"]

                    for qtype, score in acc_dict_all["perQuestionType"].items():
                        wandb_log_dict[f"Test_Qtypes/{qtype}"] = score
                    for atype, score in acc_dict_all["perAnswerType"].items():
                        if atype == "yes/no":
                            atype = "yes_no"
                        wandb_log_dict[f"Test_Atypes/{atype}"] = score

                    print(test_task, wandb_log_dict)
                    mega_log_dict[task][test_task].append(wandb_log_dict)
                    # predict_gt_dict[task][test_task].append(self.compile_preds(quesid2ans, quesid2gt))
                self.result_matrix[task][test_task] = acc_dict_all["overall"]
        # if self.args.log_all_runs:
        pred_dir = os.path.join(self.args.output, "predictions")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir, exist_ok=True)
        with open(f"{pred_dir}/{task}_acc.json", "w") as f:
            json.dump(mega_log_dict, f, indent=4)
        # with open(f"{pred_dir}/{task}_gt_pred.json", 'w') as f:
        #     json.dump(predict_gt_dict, f, indent=4)

    # def compile_preds(self, quesid2ans, quesid2gt):
    #     gt_pred_pairs = {}
    #     for key, val in quesid2ans.items():
    #         pred = val
    #         img_id, question, gt = quesid2gt[key]
    #         gt_pred_pairs[key] = [img_id, question, pred, gt]
    #     return gt_pred_pairs

    def predict(self, loader, task, dump_path=None):
        self.model.eval()
        ans_list = []
        with torch.no_grad():
            quesid2ans, quesid2gt = {}, {}
            print("Predicting")
            if args.show_train_progress:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction---")
            for i, batch in enumerate(loader):
                results = self.model.test_step(batch, task)
                pred_ans = results["pred_ans"]  # generated_sents

                ans_list.append(pred_ans)
                ques_ids = batch["question_ids"]
                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans
                    quesid2gt[qid] = batch["img_id"][0], batch["sent"][0], batch["answers"][0]
                if args.show_train_progress:
                    pbar.update(1)
            if args.show_train_progress:
                pbar.close()
        qid2ans_list = quesid2ans
        if self.verbose:
            # quesid2ans = {}
            # # for qid2ans in qid2ans_list:
            # for k, v in qid2ans.items():
            #     quesid2ans[k] = v
            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)
        return quesid2ans, quesid2gt

    def evaluate(self, loader, task, dump_path=None):
        quesid2ans, _ = self.predict(loader, task, dump_path)
        if self.verbose:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)
            topk_score = evaluator.evaluate(quesid2ans)
            acc_dict["topk_score"] = topk_score
            return acc_dict


def main_worker(gpu, args):
    # GPU is assigned
    coco_Ours = All_task
    trainer = Trainer(args, coco_Ours, train=True)
    if args.now_train:
        # List all files in the output directory and filter out non-checkpoint files
        # ckpts = [file for file in os.listdir(args.output) if file.endswith(".pth") and not os.path.isdir(os.path.join(args.output, file))]

        # # Initialize checkpoint variable
        # args.checkpoint = None

        # # # Find the latest checkpoint that matches the latest task
        # for t in reversed(All_task):  # No need to slice the list, just reverse iterate
        #     if t in ckpts:
        #         print(f"Latest checkpoint found @ {args.checkpoint}")
        #         args.checkpoint = t
        #         break

        # If a checkpoint is found, load it; otherwise, start training without loading
        if args.checkpoint != "None":
            trainer.train(load=True)
        else:
            trainer.train(load=False)

        print("#------------------ result_matrix --------------------#")
        show_results_matrix(trainer.result_matrix)
        path = args.output + "results_matrix.json"
        # save_results_matrix(trainer.result_matrix, path)
        metric_dict = evaluate_metric(trainer.result_matrix)
        print("#------  Metric  ------#")
        print("Incremental avg accuracy:", metric_dict["Incre_avg_acc"])
        print("*** Avg accuracy ***", metric_dict["Avg_acc"])
        print("Incremental avg forget:", metric_dict["Incre_avg_forget"])
        print("*** Avg forget ***", metric_dict["Avg_forget"])
        print("6Q Incremental avg accuracy:", metric_dict["Incre_avg_acc_6Q"])
        print("*** _6Q Avg accuracy ***", metric_dict["Avg_acc_6Q"])
        print("_6Q Incremental avg forget:", metric_dict["Incre_avg_forget_6Q"])
        print("*** _6Q Avg forget ***", metric_dict["Avg_forget_6Q"])

    else:
        if args.checkpoint != "None":
            task_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 9))
            task = All_task[task_idx]
            args.checkpoint = f"{args.output}/{task}_LAST"
            print(args.checkpoint)
            trainer.Test(load=True)
        else:
            trainer.Test(load=False)

        try:
            print("#------------------ Final Performance --------------------#")
            print(trainer.result_matrix["q_causal"])
            acc = 0
            for key in trainer.result_matrix["q_causal"]:
                acc += trainer.result_matrix["q_causal"][key]
            print("AP:", round(acc / 10, 4))

        except:
            pass


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    args_output_path = os.path.join(args.output, "q_recognition_LAST.pth")
    source_path = "snap/naiveblip_cl_no_ents/q_recognition_LAST.pth"
    if not os.path.exists(args_output_path):
        try:
            shutil.copyfile(source_path, args_output_path)
            print(f"Successfully copied {source_path} to {args_output_path}")
        except Exception as e:
            print(f"Failed to copy file: {e}")
    main_worker(args.local_rank, args)
