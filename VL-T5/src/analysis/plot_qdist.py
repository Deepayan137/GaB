import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sys.path.insert(0, '../')
from Question_type import Sg_task, All_task
suffix_mapping = {
        5000: '_5.0k',
        1000: '_1.0k',
        2500: '_2.5k',
        10000: '_10k',
    }

def plot_qdist(data, sub_task, dest_filepath):
    # Correct the data extraction
    data_corrected = data[sub_task]
    # Extract the methods and hyperparameter values correctly
    methods_corrected = list(data_corrected.keys())
    sorted_hyperparameters = sorted(data_corrected[methods_corrected[0]].items(), key=lambda x: x[1], reverse=True)
    hyperparameters_corrected = [k for k, v in sorted_hyperparameters]
    # methods_corrected.pop(-1)
    # Prepare data for seaborn
    data_list = []
    for method in methods_corrected:
        # keys = sorted(list(data_corrected[method].keys()))
        # for key in keys:
        for key in hyperparameters_corrected:
            if method != 'balanced':
                if method == 'real':
                    name = 'Real'
                else:
                    name = "Generated"
                try:
                    qdist = data_corrected[method][key]
                except:
                    qdist = 0.1
                data_list.append({
                    'Question Labels': key,
                    'Question Distribution': qdist,
                    'Method': name
                })

    # Convert to a DataFrame
    df = pd.DataFrame(data_list)
    # Plotting the grouped bar chart
    plt.figure(figsize=(9, 6))
    sns.barplot(x='Question Labels', y='Question Distribution', hue='Method', data=df)

    # Adding labels and title
    plt.xlabel('Question Type', fontsize=21)
    if sub_task == 'q_type':
        plt.ylabel('Question Distribution (%)', fontsize=21)
    else:
        plt.ylabel('', fontsize=12)
    name = sub_task.split('_')[-1].capitalize()
    plt.title(f'Question Distribution for task {name}', fontsize=21)
    
    plt.legend(fontsize=21, title_fontsize=21)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(dest_filepath)
    # plt.show()
    plt.close()



if __name__ == "__main__":
    task_idx = 9
    method = 'classifer'
    mem_sizes = [1000]
    # All_task = Sg_task['function']['oarlks']
    task = All_task[task_idx]
    root = 'vqacl_qdists'
    os.makedirs('plots_vqacl', exist_ok=True)
    for i in range(task_idx):
        for mem_size in mem_sizes:
            sub_task = All_task[i]
            if sub_task == 'q_type' or sub_task == 'q_location':
                mem_suffix = suffix_mapping[mem_size]
                if method == 'cluster':
                    filepath = os.path.join(root, f'vqacl_{task}{mem_suffix}_question_dist_via_clustering.json')
                else:
                    filepath = os.path.join(root, f'vqacl_{task}{mem_suffix}_question_dist.json')
                with open(filepath, 'r') as f:
                    data = json.load(f)
                filename = os.path.basename(filepath)
                if method == 'cluster':
                    dest_filename = filename.replace('question_dist_via_clustering.json', f'{sub_task}_question_dist_via_clustering.png')
                else:
                    dest_filename = filename.replace('_question_dist.json', f'_{sub_task}_question_dist.png')
                dest_filepath = os.path.join('plots_vqacl', dest_filename)
                plot_qdist(data, sub_task, dest_filepath)


