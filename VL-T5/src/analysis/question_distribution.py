import os
import torch
import argparse
import numpy as np
import json
import pickle
from collections import Counter, defaultdict
from tqdm import *
from src.analysis.question_classifier import get_embedding, QuestionTypeClassifier
import random
from sklearn.cluster import KMeans
import sys
from scipy.spatial.distance import cdist

sys.path.insert(0, "../")
from Question_type import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def label_stats(label_counts):
    total = sum(label_counts.values())
    percentages = {label: (count / total * 100) for label, count in label_counts.items()}
    return percentages


def get_question_dist(predictions):
    label_counts = Counter()
    # Increment the count for each prediction
    for prediction in predictions:
        label_counts[prediction] += 1  # Directly increment the count of each prediction
    return label_stats(label_counts)


def _load_classifier_ckpt(classifier, sub_task, name="sgvqa"):
    if name == "sgvqa":
        ckpt_path = f"../ckpt_sgvqa/{sub_task}.pth"
    else:
        ckpt_path = f"../ckpt_vqacl/{sub_task}.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading existsing checkpoint @ {ckpt_path}")
        ckpt = torch.load(f"{ckpt_path}", map_location=device)
        classifier.load_state_dict(ckpt)
    else:
        print("No ckpt found")
    return classifier


def load_gen_data(task, root, All_task):
    json_path = os.path.join(root, f"fcl_mmf_{task}_train_updated.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    task_idx = All_task.index(task)
    data_ = {}
    for i in range(task_idx):
        sub_task = All_task[i]
        if sub_task not in data_:
            data_[sub_task] = []
        for datum in data:
            question_key = f"Q_{sub_task}"
            if question_key in datum:
                data_[sub_task].append(datum[question_key])
    return data_


def load_orig_data(task, root_path, split, size=5000):
    np_path = os.path.join(root_path, f"fcl_mmf_{task}_{split}.npy")
    data = np.load(np_path, allow_pickle=True)
    random.shuffle(data)
    data = data[:size]
    data_ = {}
    data_[task] = []
    for datum in data:
        data_[task].append(datum["question"])
    return data_


def sample_by_predicted_labels(train_data, predictions, desired_counts, total_target=5000, seed=42):
    import random
    from collections import defaultdict

    # Set random seed for reproducibility
    random.seed(seed)

    # Mapping from labels to list of indices that have this label
    label_indices = defaultdict(list)
    for index, label in enumerate(predictions):
        label_indices[label].append(index)

    # Sample indices according to desired counts
    sampled_indices = []
    current_total = 0
    for label, count in desired_counts.items():
        if label in label_indices:
            if len(label_indices[label]) >= count:
                sampled = random.sample(label_indices[label], count)
            else:
                # If there aren't enough data as desired, repeat elements
                sampled = random.choices(label_indices[label], k=count)
                print(f"Warning: Not enough data for label {label}. Needed {count}, got {len(label_indices[label])}, allowing repetition.")
            sampled_indices.extend(sampled)
            current_total += count
        else:
            print(f"Warning: No data available for label {label}. Needed {count}.")

    # Collect the actual data points from the sampled indices
    sampled_data = [train_data[idx] for idx in sampled_indices]

    # Optionally, adjust the number of sampled data to match the total target size
    if len(sampled_data) > total_target:
        print(f"Warning: Sampled more data than the target ({len(sampled_data)} items), reducing to {total_target}.")
        sampled_data = sampled_data[:total_target]
    elif len(sampled_data) < total_target:
        print(f"Warning: Sampled less data than the target ({len(sampled_data)} items), expected {total_target}.")

    return sampled_data


def cluster_questions(questions, task, batch_size=32, n_clusters=10, train=False, filename=None):
    # Placeholder for all embeddings
    all_embeddings = []
    for i in trange(0, len(questions[task]), batch_size):
        batch_questions = questions[task][i : i + batch_size]
        embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
        all_embeddings.extend(embeddings)
    # Stack all embeddings into a single tensor
    embeddings_tensor = torch.stack(all_embeddings)
    embeddings_tensor = embeddings_tensor.cpu().numpy()  # Convert to numpy array for KMeans

    # Perform k-means clustering
    if train:
        if not os.path.exists(filename):
            print("Clustering and saving prototypes")
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_tensor)
            # Save the cluster centers (prototypes)
            prototypes = kmeans.cluster_centers_
            with open(filename, "wb") as file:
                pickle.dump(prototypes, file)

    if os.path.exists(filename):
        print("prototypes found loading prototypes")
        with open(filename, "rb") as file:
            prototypes = pickle.load(file)
    else:
        raise Exception(f"No Kmeans prototypes found @ {filename}")
    print("Estimating distances with ptototypes")
    # Get predicted labels for all embeddings using the loaded or newly computed prototypes
    distances = cdist(embeddings_tensor, prototypes, "euclidean")
    predicted_labels = np.argmin(distances, axis=1)
    return predicted_labels


def classify_questions(model, questions, task, batch_size=32):
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []
    # Loop through batches of questions
    for i in trange(0, len(questions[task]), batch_size):
        batch_questions = questions[task][i : i + batch_size]
        embeddings = [get_embedding(question).squeeze(0) for question in batch_questions]
        embeddings = torch.stack(embeddings)  # Convert list of tensors to a single tensor

        # Ensure the embeddings are on the same device as the model
        embeddings = embeddings.to(next(model.parameters()).device)

        with torch.no_grad():  # Disable gradient computation
            outputs = model(embeddings)  # Get the logits from the model
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            predicted_labels = torch.argmax(probabilities, dim=1)  # Get the index of the max probability
        predictions.extend(predicted_labels.cpu().numpy())
    return predictions


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--root_path", default="../datasets/npy/function/", type=str)
    parser.add_argument("--gen_path", default="../datasets/npy_no_ents/function/", type=str)
    parser.add_argument("--savepath", default="../ckpt_sgvqa", type=str)
    parser.add_argument("--strategy", type=str, default="classifier")
    parser.add_argument("--sequence", type=str, default="oarlks")
    parser.add_argument("--n_clusters", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    strategy = args.strategy
    n_clusters = 10
    sequence = args.sequence
    os.makedirs(args.savepath, exist_ok=True)
    task_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 1))
    All_task = Sg_task["function"][sequence]
    task = All_task[task_idx]  # Simplified to only run for the first task
    created = load_gen_data(task, args.gen_path, All_task)
    input_dim = 768
    hidden_dim = 256
    summary_dict = defaultdict(dict)
    # Loop over all tasks, but limit to the first for simplification
    for i, sub_task in enumerate(All_task[:task_idx]):
        output_dim = len(qtype_dict[sub_task])
        train_data = load_orig_data(sub_task, args.root_path, "train", size=20000)
        test_data = load_orig_data(sub_task, args.root_path, "train", size=5000)
        # Initialize classifier if the strategy is to classify
        if strategy == "classifier":
            classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
            classifier = _load_classifier_ckpt(classifier, sub_task)
            predictions_created = classify_questions(classifier, created, sub_task)
            predictions_test = classify_questions(classifier, test_data, sub_task)
        elif strategy == "cluster":
            filename = f"kmeans_{sub_task}_{n_clusters}.pkl" if sequence == "oarlks" else f"kmeans_{sub_task}_{sequence}.pkl"
            filename = os.path.join(args.savepath, filename)
            if not os.path.exists(filename):
                print(f"No {filename} found")
                predictions_train = cluster_questions(train_data, sub_task, train=True, filename=filename, n_clusters=n_clusters)
            predictions_test = cluster_questions(test_data, sub_task, train=False, filename=filename, n_clusters=n_clusters)
            predictions_created = cluster_questions(created, sub_task, filename=filename)
        label_counts_created = get_question_dist(predictions_created)
        label_counts_test = get_question_dist(predictions_test)

        # Store results in a more readable format
        summary_dict[sub_task] = {
            "balanced": {str(k): v for k, v in label_counts_test.items()},
            "unbalanced": {str(k): v for k, v in label_counts_created.items()},
        }

        print(f"For task {sub_task} the distribution of labels in synthetic data is {label_counts_created}")
        print(f"For task {sub_task} the distribution of labels in the test data is {label_counts_test}")

    # Save results based on strategy
    file_name = f"question_dist_via_clustering_{n_clusters}.json" if strategy == "cluster" else f"question_dist_{sequence}.json"
    dest_dir = "../metrics"
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, f"sgvqa_{task}_{file_name}"), "w") as f:
        json.dump(summary_dict, f, indent=4)
