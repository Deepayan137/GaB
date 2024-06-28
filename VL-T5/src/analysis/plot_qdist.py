import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sys.path.insert(0, '../')
from Question_type import Sg_task
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
    methods_corrected.pop(-1)
    # Prepare data for seaborn
    data_list = []
    for method in methods_corrected:
        # keys = sorted(list(data_corrected[method].keys()))
        # for key in keys:
        for key in hyperparameters_corrected:
            try:
                qdist = data_corrected[method][key]
            except:
                qdist = 0.1
            data_list.append({
                'Question Labels': key,
                'Question Distribution': qdist,
                'Method': method
            })

    # Convert to a DataFrame
    df = pd.DataFrame(data_list)
    # Plotting the grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Question Labels', y='Question Distribution', hue='Method', data=df)

    # Adding labels and title
    plt.xlabel('Question Type', fontsize=20)
    plt.ylabel('Question Distribution (%)', fontsize=16)
    # plt.title('Question Distribution Comparison of Different Methods for Each Hyperparameter')
    plt.legend(fontsize=14, title_fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(dest_filepath)
    # plt.show()
    plt.close()



if __name__ == "__main__":
    task_idx = 4
    mem_sizes = [1000, 2500, 5000, 10000]
    All_task = Sg_task['function']['oarlks']
    task = All_task[task_idx]
    root = 'qdists'
    os.makedirs('plots', exist_ok=True)
    for i in range(task_idx):
        for mem_size in mem_sizes:
            sub_task = All_task[i]
            mem_suffix = suffix_mapping[mem_size]
            filepath = os.path.join(root, f'sgvqa_{task}{mem_suffix}_question_dist.json')
            with open(filepath, 'r') as f:
                data = json.load(f)
            filename = os.path.basename(filepath)
            dest_filename = filename.replace('question_dist.json', f'{sub_task}_question_dist.png')
            dest_filepath = os.path.join('plots', dest_filename)
            plot_qdist(data, sub_task, dest_filepath)


