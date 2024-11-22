import json
import os
import time
import numpy as np
import spacy
import torch
import argparse
import random
from PIL import Image
from transformers import AutoProcessor, Blip2Config
from src.vqa_model_blip import NaiveBLIP2

# from src.param import parse_args
import sys

sys.path.insert(0, "../")
from Question_type import Sg_task

nlp = spacy.load("en_core_web_sm")
# args = parse_args()


class GenQues:
    def __init__(self, savepath=None, model=None, processor=None):
        self.savepath = savepath
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None and processor is not None:
            self.model, self.processor = model, processor
        else:
            model_name = "Salesforce/blip2-opt-2.7b"
            self.processor, self.model = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=".")
        config = Blip2Config.from_pretrained(model_name)
        model = NaiveBLIP2.from_pretrained(model_name, cache_dir=".", config=config).to(self.device)
        return processor, model

    def _load_model(self, task):
        ckpt_path = os.path.join(self.savepath, f"{task}_BEST.pth")
        print(f"Loading checkpoint from @ {ckpt_path}")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.query_tokens.data.copy_(checkpoint["model"]["query_tokens"])
            self.model.language_projection_questions.load_state_dict(checkpoint["model"]["language_projection_questions"])

    def _load_answers(self, task):
        path = f"../datasets/clove_stats/{task}_answer_stats.json"
        with open(path, "r") as f:
            answer_dict = json.load(f)

        # Determine the number of top answers to keep based on the task type
        if task in ["object", "attribute"]:
            k = 150
        elif task == "relation":
            k = 20
        elif task == "logical":
            k = 2

        # Sort the dictionary by their frequencies (values) in descending order
        sorted_answers = sorted(answer_dict, key=answer_dict.get, reverse=True)

        # Store only the top k answers
        self.answer_list = sorted_answers[:k]

    def _load_qtype_stats(self, task):
        if task == "object":
            elements = [0, 1, 2]  # Corrected
            weights = [0.4612, 0.3348, 0.204]  # Directly use given percentages
        elif task == "attribute":
            elements = [0, 1, 4, 5, 6, 7, 8, 11]
            weights = [0.1788, 0.2464, 0.1112, 0.0148, 0.054, 0.0796, 0.2892, 0.026]  # Corrected weights
        elif task == "relation":
            elements = [0, 1, 5, 7, 13]
            weights = [0.1975, 0.2329, 0.2563, 0.2311, 0.0822]  # Adjusted for sum = 1
        elif task == "logical":
            elements = [0, 2, 3, 6, 10]
            weights = [0.08, 0.32, 0.1344, 0.232, 0.2336]  # Adjusted for sum = 1
        weights = [w / sum(weights) for w in weights]  # Normalize weights to sum to 1
        sampled_element = random.choices(elements, weights=weights, k=1)[0]
        return sampled_element

    def clean_question(self, output):
        # Define the keyword to identify the part of the string to remove
        keyword = "Question:"
        output = output.strip()
        # Split the output on the keyword
        parts = output.split(keyword)

        # Determine the part to use
        remaining_part = parts[-1] if len(parts) > 1 else output

        # Check if the remaining part starts with 'by' and handle accordingly
        remaining_part = remaining_part.strip().lower()
        if remaining_part.startswith("by") or remaining_part.startswith("for"):
            # Skip the first word
            return " ".join(remaining_part.split(" ")[1:])
        else:
            # Return the trimmed text if it doesn't start with 'by'
            return remaining_part

    def _batchify(self, data, batch_size, task):
        num_batches = (len(data) + batch_size - 1) // batch_size
        image_dir = "gvqa"

        def batch_generator():
            for i in range(num_batches):
                data_subset = data[i * batch_size : (i + 1) * batch_size]
                images, prompts = [], []
                for _d in data_subset:
                    img_name = f"{_d['image_id']}.jpg"
                    img_path = os.path.join(f"../datasets/{image_dir}", img_name)
                    image = Image.open(img_path).convert("RGB")
                    # if method == 'lamol':
                    # 	prompt = f'{task}'
                    images.append(image)
                    # prompts.append(prompt)

                batch_inputs = self.processor(images, return_tensors="pt", padding=True, truncation=True).to(self.device)
                yield {"data": data_subset, "inputs": batch_inputs}

        return batch_generator(), num_batches

    def post_process_answer(self, generated_ids, task):
        questions = generated_ids["questions"]
        pairs = []

        for question in questions:
            try:
                question, answer = question.split("?")
                if "question:" in question:
                    question = question.split("question:")[-1]
                    question = question + "?"
                # if method == 'lamol':
                # 	question = f'{task}:'+question
                if "answer:" in answer:
                    answer = answer.split("answer:")[-1].strip()
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                if len(answer.split(" ")) > 2:
                    answer = " ".join(answer.split(" ")[:2])
            except:
                import pdb

                pdb.set_trace()
                print("Oops")

            pairs.append((question, answer))
        return pairs

    def generate_qa(self, data, task):
        max_new_tokens = 40
        batch_generator, total_batches = self._batchify(data, 32, task)
        pairs = []
        for i, batch in enumerate((batch_generator)):
            data_subset, inputs = batch["data"], batch["inputs"]
            try:
                # generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                # 	num_beams=5, temperature=0.9, do_sample=True, repetition_penalty=1.2)
                generated_ids = self.model.get_questions(inputs, max_new_tokens=max_new_tokens)
                import pdb

                pdb.set_trace()
                qa_pairs = self.post_process_answer(generated_ids, task)
                pairs.extend(zip(data_subset, qa_pairs))
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                continue
            print(f"{i+1} out of {total_batches} completed")
        return pairs


def create_rehearsal_data(args, task_idx):
    path = args.path
    # method = args.method
    sequence = args.sequence
    task = Sg_task["function"][sequence][task_idx]
    gen_ques = GenQues(savepath=args.savepath)
    mem_size = args.mem_size
    split = int(mem_size / task_idx)
    fname = f"fcl_mmf_{task}_train.npy"
    source = os.path.join(path, fname)
    data = np.load(source, allow_pickle=True)
    incorrect = 0
    new_data = []
    for i in range(task_idx):
        qg_task = Sg_task["function"][sequence][i]
        start_idx = i * split
        end_idx = start_idx + split
        print(f"Now task is {task} and question generation will be from {qg_task}")
        print(f"start idx: {start_idx}")
        print(f"end idx: {end_idx}")
        data_subset = data[start_idx:end_idx]
        gen_ques._load_model(qg_task)
        print(f"Number of samples: {len(data_subset)}")
        start = time.time()
        pairs = gen_ques.generate_qa(data_subset, qg_task)
        print(f"Time taken for generation:{time.time()-start}")
        for i, pair in enumerate(pairs):
            try:
                _d, (questions, answers) = pair
                _d[f"Q_{qg_task}"] = questions
                _d[f"A_{qg_task}"] = answers
                new_data.append(_d)
            except Exception as e:
                print(f"Unexpected error at index {i}: {e}")
    print(f"Incorrect: {incorrect} in {len(data)} samples")
    dest_fname = os.path.join(args.dest_dir, f"fcl_mmf_{task}_train.json")
    with open(dest_fname, "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser("")
        parser.add_argument("--path", default="../datasets/npy/function", type=str)
        parser.add_argument("--img_path", default="../datasets/gvqa", type=str)
        parser.add_argument("--savepath", default="snap/naiveblip_sgvqa_qg_lkora", type=str)
        parser.add_argument("--mem_size", type=int, default=20000)
        parser.add_argument("--sequence", type=str, default="oarlks")
        parser.add_argument("--dest_dir", default="../datasets/npy_test/function/")
        return parser.parse_args()

    args = get_args()
    task_idx = 4
    dest_dir = args.dest_dir
    create_rehearsal_data(args, task_idx)
