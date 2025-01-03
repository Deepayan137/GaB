import os
import json
import random
import sys


sys.path.insert(0, "../")
from Question_type import *

from tqdm import *

import warnings

warnings.filterwarnings("ignore")
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Your code follows here...

MODEL_DIR = "./llama-2-7b-chat-hf"
All_task = [
    "q_recognition",
    "q_location",
    "q_judge",
    "q_commonsense",
    "q_count",
    "q_action",
    "q_color",
    "q_type",
    "q_subcategory",
    "q_causal",
]


def build_data_info_path(args, scenario_dir, tsk):
    # Define the suffix based on M
    if args.method == "lamol":
        root = "../datasets/vqa/Partition_Q_V2_lamol/"
        fname = f"karpathy_train_{tsk}.json"
        data_info_path = os.path.join(root, fname)
    else:
        suffix_mapping = {
            1000: "_1.0k",
            5000: "_5.0k",
            2500: "_2.5k",
            10000: "_10k",
        }

        # Determine the balance type
        if args.balance_strategy == "classifier":
            balance_type = "balanced"
        elif args.balance_strategy == "cluster":
            balance_type = f"cluster_balanced_{args.n_clusters}"
        else:
            balance_type = "unbalanced"

        # Get the appropriate suffix for the given M, default to an empty string if not found
        suffix = suffix_mapping.get(args.m_size, "")

        # Construct the file path
        file_name = f"karpathy_train_{tsk}_{balance_type}{suffix}.json"
        data_info_path = os.path.join(scenario_dir, file_name)

    return data_info_path


def get_memory_data(args, task_idx, each_memory, Examplar_set, model, processor):
    print("Welcome to the rehearsal memory module")
    if args.use_gen_data:
        print("We will use Synthetic QA pairs")
        task = All_task[task_idx]
        dest = f"../datasets/vqa/Partition_Q_V2_{args.method}/"
        if not os.path.exists(f"{dest}/karpathy_train_{task}.json"):
            print(f"Synthetic QA pairs not found so creating  for task {task}")
            create_rehearsal_data(args, task_idx, dest, model=model, processor=processor, savepath=None)
        each_memory = args.m_size
        Examplar_set = {"G1": [], "G2": [], "G3": [], "G4": [], "G5": []}
        scenario_dir = dest
        data_info_path = build_data_info_path(args, scenario_dir, All_task[task_idx])
    else:
        print("Loading real QA pairs from previous tasks")
        dest = "../datasets/vqa/Partition_Q_V2/"
        data_info_path = f"{dest}/karpathy_train_" + f"{All_task[task_idx - 1]}.json"
    print(f"Loading data from {data_info_path}")
    with open(data_info_path) as f:
        data_info_dicts = json.load(f)
    random.shuffle(data_info_dicts)
    if args.use_class_hierarchy:
        each_memory_for_cate = int(each_memory / len(Category_splits))
        for cate in Category_splits:
            num = 0
            if not args.use_gen_data:
                Examplar_set[cate].append([])
            for _d in data_info_dicts:
                img_id = _d["img_id"]
                if img_id in ImgId_cate_map:
                    if ImgId_cate_map[img_id] in Category_splits[cate]:
                        if not args.use_gen_data:
                            Examplar_set[cate][task_idx - 1].append(_d)
                        else:
                            Examplar_set[cate].append(_d)
                        num += 1
                        if num >= each_memory_for_cate:
                            break
        print("Load from Partition_Q_v3......")
        if not args.use_gen_data:
            for cate in Category_splits:
                for i in range(task_idx):
                    Examplar_set[cate][i] = Examplar_set[cate][i][:each_memory_for_cate]
            All_examplar = []
            for E_set in Examplar_set:
                for task_set in Examplar_set[E_set]:
                    All_examplar += task_set
        else:
            All_examplar = []
            for E_set in Examplar_set:
                All_examplar += Examplar_set[E_set]
        new_data = []
        for datum in All_examplar:
            new_datum = {}
            for k, v in datum.items():
                if k.startswith("Q_"):
                    new_datum["Q"] = datum[k]
                elif k.startswith("A_"):
                    new_datum["A"] = datum[k]
                else:
                    new_datum[k] = datum[k]
            new_data.append(new_datum)
        All_examplar = new_data
    else:
        All_examplar = data_info_dicts[:each_memory]
    print("# The size of the cate Memory:", len(All_examplar))
    return All_examplar, Examplar_set


def create_rehearsal_data(args, task_idx, dest_dir, savepath=None, model=None, processor=None):
    os.makedirs(dest_dir, exist_ok=True)
    task = All_task[task_idx]
    fname = f"karpathy_train_{task}.json"
    dest = os.path.join(f"{dest_dir}", fname)
    path = "../datasets/vqa/Partition_Q_V2/"
    from src.analysis.vqacl_gen_ques import _load_data, GenQues

    with open(f"metrics/{task}_question_dist.json", "r") as f:
        desired_counts = json.load(f)

    gen_ques = GenQues(savepath=args.output, model=model, processor=processor)
    mem_size = 20000
    if args.method == "lamol":
        mem_size = 5000
        lamol_task = All_task[task_idx - 1]
        gen_ques._load_model(lamol_task)
    data = _load_data(path, task_idx, mem_size)
    incorrect = 0
    new_data = []
    if task != "q_causal":
        partition = int(mem_size / task_idx)
    else:
        partition = int(len(data) / task_idx)
    for i in range(task_idx):
        qg_task = All_task[i]
        start_idx = i * partition
        end_idx = start_idx + partition
        print(f"Now task is {task} and question generation will be from {qg_task}")
        print(f"start idx: {start_idx}")
        print(f"end idx: {end_idx}")
        data_subset = data[start_idx:end_idx]
        if args.method != "lamol":
            gen_ques._load_model(qg_task)
        print(f"Number of samples: {len(data_subset)}")
        for _d in tqdm(data_subset):
            img_name = f"{_d['img_id']}.jpg"
            split = "train" if "train2014" in img_name else "val"
            img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
            pairs = gen_ques.inference_qa(img_path, _d, qg_task, desired_counts[qg_task], args.method, self_answer=False)
            if pairs != None and len(pairs) > 0:
                questions, answers = zip(*pairs)
                if "" not in answers or "" not in answers:
                    _d[f"Q_{qg_task}"] = questions[0]
                    _d[f"A_{qg_task}"] = answers[0]
                    new_data.append(_d)
            else:
                incorrect += 1
    print(f"Incorrect: {incorrect} in {len(data)} samples")
    with open(dest, "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    from src.param import parse_args

    args = parse_args()
    task_idx = 2
    args.method = "lamol"
    args.output = "snap/naiveblip_cl_lamol/"
    dest_dir = f"../datasets/vqa/Partition_Q_V2_{args.method}/"
    create_rehearsal_data(args, task_idx, dest_dir, savepath=args.output)
