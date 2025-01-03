import os
import argparse
import json
from tqdm import *
import torch
import random
import sys

sys.path.insert(0, "../")
from Question_type import *

from tqdm import *

device = "cuda" if torch.cuda.is_available() else "cpu"

from src.analysis.question_classifier import QuestionTypeClassifier
from src.analysis.question_distribution import (
    get_question_dist,
    classify_questions,
    cluster_questions,
    sample_by_predicted_labels,
    _load_classifier_ckpt,
)


def unbalanced_data(data, task, split, All_task):
    # Define the root path for the captions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Find the task index and prepare to handle the data up to that task
    task_idx = All_task.index(task)
    final_data = []
    # input_dim = 768
    # hidden_dim = 256
    # Iterate through each task up to the current task index
    sub_task_questions = {}
    for i in range(task_idx):
        sub_task = All_task[i]
        sub_task_questions[sub_task] = []
        # output_dim = len(qtype_dict[sub_task])
        # classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
        # classifier = _load_classifier_ckpt(classifier, sub_task)
        question_key = f"Q_{sub_task}"
        answer_key = f"A_{sub_task}"

        # Filter out data items that don't have both the question_key and answer_key
        sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
        # Iterate over each filtered data item to process questions and answers
        new_data = []
        for datum in sub_data:
            sub_task_questions[sub_task] = []
            new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
            questions = datum.get(question_key, [])
            answers = datum.get(answer_key, [])
            # Create QA pairs excluding 'room' from answers
            qa_pairs = [(q, a) for q, a in zip(questions, answers)]

            # Select a random QA pair if available and add to new_data
            if qa_pairs:
                question, answer = random.choice(qa_pairs)
                new_datum[f"Q_{sub_task}"] = question
                new_datum[f"A_{sub_task}"] = " ".join(answer.split(" ")[:2])
                sub_task_questions[sub_task].append(question)
                new_data.append(new_datum)
        random.shuffle(new_data)
        new_data = new_data[:split]
        final_data.extend(new_data)

    return final_data


def balanced_data_via_clustering(data, task, split, All_task, name="sgvqa", sequence="oarlks", n_clusters=10, seed=42):
    if name == "sgvqa":
        if sequence == "oarlks":
            fpath = f"../metrics/sgvqa_{task}_question_dist_via_clustering_{n_clusters}.json"
        else:
            fpath = f"../metrics/sgvqa_{task}_question_dist_via_clustering_{sequence}.json"
        with open(fpath, "r") as f:
            desired_counts = json.load(f)
    else:
        with open(f"../metrics/{task}_question_dist_via_clustering_{n_clusters}.json", "r") as f:
            desired_counts = json.load(f)
    # Define the root path for the captions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Find the task index and prepare to handle the data up to that task
    task_idx = All_task.index(task)
    final_data = []
    # Iterate through each task up to the current task index
    sub_task_questions = {}
    for i in range(task_idx):
        sub_task = All_task[i]
        sub_task_questions[sub_task] = []
        desired_task_counts = desired_counts[sub_task]["balanced"]
        desired_task_counts = {int(k): int(v * split / 100.0) for k, v in desired_task_counts.items()}
        if name == "sgvqa":
            output_dim = len(qtype_dict[sub_task])
        else:
            output_dim = len(cat_dict_vqacl[sub_task])
        question_key = f"Q_{sub_task}"
        answer_key = f"A_{sub_task}"

        # Filter out data items that don't have both the question_key and answer_key
        sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
        count = 0
        # Iterate over each filtered data item to process questions and answers
        new_data = []
        for datum in sub_data:
            new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
            questions = datum.get(question_key, [])
            answers = datum.get(answer_key, [])
            # Create QA pairs excluding 'room' from answers
            qa_pairs = [(q, a) for q, a in zip(questions, answers)]

            # Select a random QA pair if available and add to new_data
            if qa_pairs:
                question, answer = random.choice(qa_pairs)
                new_datum[f"Q_{sub_task}"] = question
                new_datum[f"A_{sub_task}"] = " ".join(answer.split(" ")[:2])
                sub_task_questions[sub_task].append(question)
                new_data.append(new_datum)
        if name == "sgvqa":
            filename = f"../ckpt_sgvqa/kmeans_{sub_task}_{n_clusters}.pkl" if sequence == "oarlks" else f"ckpt/kmeans_{sub_task}_{sequence}.pkl"
        else:
            filename = f"../ckpt_vqacl/kmeans_{sub_task}_{n_clusters}.pkl"
        predictions = cluster_questions(sub_task_questions, sub_task, n_clusters=n_clusters, train=False, filename=filename)
        new_data = sample_by_predicted_labels(new_data, predictions, desired_task_counts, total_target=split, seed=seed)
        new_questions = {}
        new_questions[sub_task] = [datum[f"Q_{sub_task}"] for datum in new_data]
        ques_preds = cluster_questions(new_questions, sub_task, filename=filename, n_clusters=n_clusters)
        label_stats = get_question_dist(ques_preds)
        print(label_stats)
        final_data.extend(new_data)

    return final_data


def balanced_data_via_classifier(data, task, split, All_task, name="sgvqa"):
    if name == "sgvqa":
        with open(f"../metrics/sgvqa_{task}_question_dist.json", "r") as f:
            desired_counts = json.load(f)
    else:
        with open(f"../metrics/{task}_question_dist.json", "r") as f:
            desired_counts = json.load(f)
    # Define the root path for the captions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Find the task index and prepare to handle the data up to that task
    task_idx = All_task.index(task)
    final_data = []
    input_dim = 768
    hidden_dim = 256
    # Iterate through each task up to the current task index
    sub_task_questions = {}
    for i in range(task_idx):
        sub_task = All_task[i]
        sub_task_questions[sub_task] = []
        desired_task_counts = desired_counts[sub_task]["balanced"]
        desired_task_counts = {int(k): int(v * split / 100.0) for k, v in desired_task_counts.items()}
        if name == "sgvqa":
            output_dim = len(qtype_dict[sub_task])
        else:
            output_dim = len(cat_dict_vqacl[sub_task])
        classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
        classifier = _load_classifier_ckpt(classifier, sub_task, name=name)
        question_key = f"Q_{sub_task}"
        answer_key = f"A_{sub_task}"

        # Filter out data items that don't have both the question_key and answer_key
        sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
        count = 0
        # Iterate over each filtered data item to process questions and answers
        new_data = []
        for datum in sub_data:
            new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
            questions = datum.get(question_key, [])
            answers = datum.get(answer_key, [])
            # Create QA pairs excluding 'room' from answers
            qa_pairs = [(q, a) for q, a in zip(questions, answers)]

            # Select a random QA pair if available and add to new_data
            if qa_pairs:
                question, answer = random.choice(qa_pairs)
                new_datum[f"Q_{sub_task}"] = question
                new_datum[f"A_{sub_task}"] = " ".join(answer.split(" ")[:2])
                sub_task_questions[sub_task].append(question)
                new_data.append(new_datum)
        predictions = classify_questions(classifier, sub_task_questions, sub_task)
        new_data = sample_by_predicted_labels(new_data, predictions, desired_task_counts, total_target=split)
        new_questions = {}
        new_questions[sub_task] = [datum[f"Q_{sub_task}"] for datum in new_data]
        ques_preds = classify_questions(classifier, new_questions, sub_task)
        label_stats = get_question_dist(ques_preds)
        print(label_stats)
        final_data.extend(new_data)

    return final_data


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--root_path", default="../datasets/npy_no_ents/function", type=str)
    parser.add_argument("--savepath", default="../ckpt_vqacl", type=str)
    parser.add_argument("--strategy", type=str, default="classifier")
    parser.add_argument("--sequence", type=str, default="oarlks")
    parser.add_argument("--mem_size", type=int, default=5000)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    strategy = args.strategy
    n_clusters = args.n_clusters
    mem_size = args.mem_size
    task_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 2))
    sequence = args.sequence
    All_task = Sg_task["function"][sequence]
    # for mem_size in mem_sizes:
    print(f"Memory Size is {mem_size}")
    # for task_idx in range(1, 5):
    task = All_task[task_idx]  # Task indices adjusted for 0-based indexing
    method = "no_ents"
    split = int(mem_size / task_idx)
    cap_root = args.root_path
    if sequence == "oarlks":
        json_path = os.path.join(cap_root, f"fcl_mmf_{task}_train_updated.json")
    else:
        json_path = os.path.join(cap_root, f"fcl_mmf_{task}_train_updated_{sequence}.json")

    print(f"Loading data from {json_path}")
    with open(json_path, "r") as file:
        data = json.load(file)
    print(f"Number of data points present in original data: {len(data)}")
    if strategy == "classifier":
        rehearsal_data = balanced_data_via_classifier(data, task, split, All_task)
        balance_status = "balanced"
    elif strategy == "cluster":
        rehearsal_data = balanced_data_via_clustering(data, task, split, All_task, sequence=sequence, n_clusters=n_clusters, seed=args.seed)
        balance_status = f"cluster_balanced_{n_clusters}"
    else:
        rehearsal_data = unbalanced_data(data, task, split, All_task)
        balance_status = "unbalanced"
    print(f"No. of samples present in {task} data file: {len(rehearsal_data)}")
    savepath_suffix = f"{float(mem_size/1000)}k" if mem_size != 10000 else "10k"
    if sequence == "oarlks":
        savepath = json_path.replace("_updated.json", f"_{balance_status}_{savepath_suffix}.json")
    else:
        savepath = json_path.replace(f"_updated_{sequence}.json", f"_{balance_status}_{savepath_suffix}_{sequence}.json")

    print(f"Saving data @ {savepath}")
    with open(savepath, "w") as f:
        json.dump(rehearsal_data, f, indent=4)

        # print("####### Finished #######")
