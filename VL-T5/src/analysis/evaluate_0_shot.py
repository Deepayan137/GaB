import sys
sys.path.append("src")
from sgvqa_data_blip import SGVQA, SGVQAEvaluator 
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
    root = '/leonardo_scratch/fast/IscrC_CLRT-VLM/llama/sgvqa_generated_ans'
    
    tasks = ["object", "attribute", "relation","logical", "knowledge", "scenetext"]
    files = [f"{t}_gen_qa_pairs.json" for t in tasks] 

    files = [os.path.join(root, fi) for fi in files] 
    acc_dict = {}
    for fi, task in zip(files, tasks):
        data_df = load_qa([fi])
        questions_id = list(data_df['question_id'])
        # import pdb;pdb.set_trace()
        answers = list(data_df['llama_ans'])
        all_answers = list(data_df['answers'])
        quesid2ans = {}
        for i, qid in enumerate(questions_id):
            quesid2ans[qid] = (answers[i], all_answers[i])
        # quesid2ans = dict(zip(questions_id, answers))
    
        dataset = SGVQA(task=task, scenario='function')    
        evaluator = SGVQAEvaluator()
    
        pred = evaluator.evaluate_raw(quesid2ans) 
        acc_dict.update({task: pred['overall']})
    print(acc_dict)