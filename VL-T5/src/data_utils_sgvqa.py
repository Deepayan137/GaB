import os
import json
import sys
from tqdm import *
import torch

from src.sgvqa_data_blip import get_loader_memory

sys.path.insert(0, "../")
from Question_type import *


class SGVQA_QAGen:
    def __init__(self, args, model, processor):
        self.args = args
        self.device = model.device
        self.savepath = args.output
        self.model = model
        self.processor = processor
        self.model.eval()

    def _load_model(self, task):
        ckpt = torch.load(os.path.join(self.savepath, f"{task}_LAST.pth"))
        print(f"question and answer gen projection head loaded for task {task} from {self.savepath}")
        self.model.language_projection_questions.load_state_dict(ckpt["model"]["language_projection_questions"])

    def generate(self, data, task, batch_size=32):
        entries = {}
        count = 0
        self._load_model(task)
        self.args.use_gen_data = False
        loader = get_loader_memory(self.args, data, batch_size=batch_size)
        for i, batch in enumerate(tqdm(loader)):
            outputs = self.model.get_questions(batch)
            qids = batch["question_ids"]
            generated_qa = outputs["questions"]
            questions = [f"{item.split('?')[0]}?" for item in generated_qa]
            cap_answers = [item.split("?")[1] if len(item.split("?")) > 1 else "" for item in generated_qa]
            sents = [f"Question: {question} Answer:" for question in questions]
            input_ids = self.processor.tokenizer(text=sents, max_length=20, truncation=True, padding=True, return_tensors="pt")
            batch["input_ids"] = input_ids["input_ids"]
            outputs = self.model.test_step(batch, task)
            answers = outputs["pred_ans"]
            # import pdb;pdb.set_trace()
            try:
                for i, qid in enumerate(qids):
                    entries[qid] = (questions[i], answers[i], cap_answers[i])
                count += len(qids)
            except Exception as e:
                print(f"Err processing batch {i+1}: {e}")
                continue
        print("Setting the old flag back")
        self.args.use_gen_data = True  # restoring the gen data flag, this is only temporary, need to find an efficient soln
        # self.model.train()
        return entries


def create_rehearsal_data(args, task_idx, model, processor, All_examplar, dest):
    os.makedirs(dest, exist_ok=True)
    qagen = SGVQA_QAGen(args, model, processor)
    task = Sg_task["function"]["oarlks"][task_idx]
    split = int(5000 / task_idx)
    new_data = []
    total = 0
    for i in range(task_idx):
        qg_task = Sg_task["function"]["oarlks"][i]
        start_idx = i * split
        end_idx = start_idx + split
        print(f"Now task is {task} and question generation will be from {qg_task}")
        print(f"start idx: {start_idx}")
        print(f"end idx: {end_idx}")
        data_subset = All_examplar[start_idx:end_idx]
        print(f"Number of samples: {len(data_subset)}")
        entries = qagen.generate(data_subset, qg_task, batch_size=32)
        count = 0
        for i, examplar in enumerate(data_subset):
            new_examplar = {k: v for k, v in examplar.items() if not k.startswith(("Q_", "A_"))}
            qid = new_examplar["question_id"]
            question, answer, caption_answer = (entry.strip() for entry in entries[qid])
            if caption_answer:
                answer = answer or "not sure"
                # Only process further if both answers are not "not sure" and are identical
                # if answer and caption_answer != "not sure" and caption_answer == answer:
                new_examplar[f"Q_{qg_task}"] = question
                new_examplar[f"A_self_{qg_task}"] = answer.strip()
                new_examplar[f"A_cap_{qg_task}"] = caption_answer.strip()
                new_data.append(new_examplar)
                total += 1
                count += 1

    with open(f"fcl_mmf_{task}_train.json", "w") as json_file:
        json.dump(new_data, json_file, indent=4)
    print("Finished\n")
    print(f"Total number of samples:{total}")
    return new_data


if __name__ == "__main__":
    import torch
    from transformers import Blip2Config

    # from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration
    from src.vqa_model_blip import NaiveBLIP2
    from transformers import AutoProcessor
    from src.param import parse_args

    args = parse_args()
    scenario_dir = "../datasets/npy_ents/function"
    data_info_path = os.path.join(scenario_dir, "fcl_mmf_attribute_train.json")
    with open(data_info_path, "r") as file:
        All_examplar = json.load(file)
    backbone = "Salesforce/blip2-opt-2.7b"
    config = Blip2Config.from_pretrained(backbone)
    processor = AutoProcessor.from_pretrained(backbone)
    model = NaiveBLIP2.from_pretrained(
        backbone,
        config=config,
        device_map="auto",
    )
    device = "cuda"
    model = model.to(device)
    savepath = "snap/naiveblip_sgvqa_rev_cap/object_BEST.pth"
    ckpt = torch.load(savepath)
    model.query_tokens.data.copy_(ckpt["model"]["query_tokens"])
    model.language_projection_answers.load_state_dict(ckpt["model"]["language_projection_answers"])
    model.language_projection_questions.load_state_dict(ckpt["model"]["language_projection_questions"])
    args.output = "snap/naiveblip_sgvqa_rev_cap/"
    args.backbone = backbone
    dest = "../datasets/npy_ents/function/"
    new_data = create_rehearsal_data(args, 1, model, processor, All_examplar, dest)
    print(f"Length of new data is: {len(new_data)}")
