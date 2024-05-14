import sys
sys.path.append("src")
from vqa_data_blip import VQADataset, VQAEvaluator 
from pathlib import Path
import os
import json
import pandas as pd

def load_qa(files):
    data_dfs = [] 
    for fi in files:
        with open(fi, 'r') as fp:
            file_dict = json.load(fp)
            file_df = pd.DataFrame.from_dict(file_dict)
            data_dfs.append(file_df)
    data_df = pd.concat(data_dfs)
    return data_df


if __name__ == '__main__':
    root = Path('/home/dtalon/dev/vqacl/out/LLaMA-0shot-answers')
    
    tasks = ["q_recognition",
            "q_location",
            "q_judge",
            "q_commonsense",
            "q_count",
            "q_action",
            "q_color",
            "q_type",
            "q_subcategory",
            "q_causal"
            ]
    files = [f"{t}_gen_qa_pairs.json" for t in tasks] 

    files = [root / fi for fi in files] 
    acc_dict = {}
    for fi, task in zip(files, tasks):
        data_df = load_qa([fi])
        questions_id = list(data_df['question_id'])
        answers = list(data_df['llama_ans'])
        quesid2ans = dict(zip(questions_id, answers))
    
        dataset = VQADataset(splits=f"Partition_Q/karpathy_test_{task}")    
        evaluator = VQAEvaluator(dataset=dataset)
    
        pred = evaluator.evaluate_raw(quesid2ans) 
        acc_dict.update({task: pred['overall']})
    print(acc_dict)