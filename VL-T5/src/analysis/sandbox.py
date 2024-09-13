# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# from torch.utils.data import Sampler
# import numpy as np
# from itertools import cycle, islice

# class SingleDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# class MultiTaskDataset(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets
#         self.dataset_lengths = [len(d) for d in datasets]
#         self.total_length = sum(self.dataset_lengths)
#         self.combined = []
#         for dataset in datasets:
#             self.combined.extend(dataset)

#     def __len__(self):
#         return self.total_length

#     def __getitem__(self, idx):
#         return self.combined[idx]

#     @staticmethod
#     def collate_fn(batch):
#         data = [item for item in batch]
#         return torch.stack(data)

# if __name__ == "__main__":
#     batch_size = 30
#     dataset1 = [torch.tensor([i]) for i in range(100)]
#     train_dataset = SingleDataset(dataset1)
#     train_loader = DataLoader(train_dataset, batch_size=32)
#     num_iterations = len(train_loader)  # Should be 32
#     dataset2 = [torch.tensor([i + 100]) for i in range(200)]
#     dataset3 = [torch.tensor([i + 300]) for i in range(150)]
#     dataset4 = [torch.tensor([i + 450]) for i in range(150)]

#     datasets = [dataset2, dataset3, dataset4]
#     combined_dataset = MultiTaskDataset(datasets)
#     lengths = combined_dataset.dataset_lengths
#     adjusted_indices = []
#     for _ in range(num_iterations):
#         indices = [np.random.choice(length, (batch_size//len(lengths)), replace=False) for length in lengths]
#         offsets = [sum(lengths[:i]) for i in range(len(lengths))]
#         for offset, indices_for_dataset in zip(offsets, indices):
#             adjusted_indices.extend([index + offset for index in indices_for_dataset])
    
#     subset = Subset(combined_dataset, adjusted_indices)
#     memory_loader = DataLoader(subset, batch_size=30, shuffle=True)

    

#     # Cycle through the memory loader but limit to the same number of iterations
#     # limited_memory_loader = islice(cycle(memory_loader), num_iterations)
    
#     now_loader = zip(train_loader, cycle(memory_loader))
#     # # Train loop
#     for now_batch in now_loader:
#         if len(now_batch) == 2:
#             batch, mem_batch = now_batch
#         else:
#             batch = now_batch
#             mem_batch = None
#         import pdb;pdb.set_trace()


# import os
# import sys
# import numpy as np
# sys.path.insert(0, '../')
# from Question_type import qtype_dict, Sg_task
# import json

# path='../datasets/npy/function/'
# All_task=Sg_task['function']['oarlks']

# QtypeDict = {}
# for task in All_task[:-2]:
#     np_path = os.path.join(path, f'fcl_mmf_{task}_val.npy')
#     data=np.load(np_path, allow_pickle=True)
#     for datum in data:
#         qtype = datum['raw_question_type']
#         qid = datum['question_id']
#         if qtype != 'none':
#             QtypeDict[qid] = str(qtype)
# with open('../datasets/SGVQA_Qtype_map.json', 'w') as f:
#     json.dump(QtypeDict, f, indent=4)


# import os
# import json
# import numpy as np
# np_path=('../datasets/npy_no_ents/function/fcl_mmf_logical_train_5.0k_rolak.json')
# with open(np_path, 'r') as f:
#     data = json.load(f)

# count_log = 0
# count_att = 0
# count_rel = 0
# count_obj = 0
# count_kno = 0
# for datum in data:
#     for key in datum.keys():
#         if key == 'Q_logical':
#             count_log += 1
#         if key == 'Q_attribute':
#             count_att += 1
#         if key == 'Q_relation':
#             count_rel += 1
#         if key == 'Q_object':
#             count_obj += 1
#         if key == 'Q_knowledge':
#             count_kno += 1
# print(f"Logical is {count_log}")
# print(f"Attribute is {count_att}")
# print(f"Relation is {count_rel}")
# print(f"Object is {count_obj}")
# print(f"knowledge is {count_kno}")
# print(len(data))
# all_questions = count_log + count_att + count_rel + count_obj + count_kno
# print(f"All questions added {all_questions}")

# import os
# import json
# from collections import Counter, defaultdict
# import sys
# sys.path.insert(0, '../')
# from Question_type import *
# # Assuming these methods are defined correctly in the imported module
# from src.analysis.vqacl_question_distribution2 import build_data_info_path, load_gen_data, load_orig_data

# def get_top_questions(questions):
#     question_counter = Counter(questions)
#     # Retrieve the top 10 most common questions
#     top_questions = question_counter.most_common(10)
#     return top_questions

# def process_tasks(root, task, mem_sizes, strategy):
#     summary_dict = defaultdict(dict)
#     for mem_size in mem_sizes:
#         balanced_json_path = build_data_info_path(root, task, mem_size, strategy)
#         print(f"Loading data from {balanced_json_path}")
#         balanced = load_gen_data(task, balanced_json_path)

#         unbalanced_json_path = build_data_info_path(root, task, mem_size, 'none')
#         print(f"Loading data from {unbalanced_json_path}")
#         unbalanced = load_gen_data(task, unbalanced_json_path)

#         orig_data = load_orig_data(task, 'train', size=mem_size)

#         for i, sub_task in enumerate(All_task[:task_idx]):
#             summary_dict[sub_task]['real'] = get_top_questions(orig_data[sub_task])
#             summary_dict[sub_task]['balanced'] = get_top_questions(balanced[sub_task])
#             summary_dict[sub_task]['unbalanced'] = get_top_questions(unbalanced[sub_task])

#         suffix = f"_{mem_size}k" if mem_size in [1000, 2500, 10000] else ''
#         file_name = "question_counter_via_clustering.json" if strategy == 'cluster' else "question_counter.json"
#         file_path = os.path.join("vqacl_qdists", f"vqacl_{task}{suffix}_{file_name}")
        
#         with open(file_path, 'w') as f:
#             json.dump(summary_dict, f, indent=4)
#         print(f"Data saved to {file_path}")

# if __name__ == "__main__":
#     task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 9))
#     task = All_task[task_idx]  # This should be defined in the imported modules or earlier in the script
#     root = "../datasets/vqa/Partition_Q_V2_no_ents/"
#     mem_sizes = [1000, 10000]
#     strategy = 'cluster'
    
#     process_tasks(root, task, mem_sizes, strategy)


# import pandas as pd
# from io import StringIO

# data = """
# ,q_recognition,q_location,q_judge,q_commonsense,q_count,q_action,q_color,q_type,q_subcategory,q_causal
# q_recognition,34.75,28.2,7.72,8.36,0.96,21.06,2.04,21.89,31.85,27.28
# q_location,20.54,26.68,2.41,3.16,0.25,5.81,0.39,16.76,28.17,21.55
# q_judge,48.23,37.91,84.69,81.81,7.04,84.44,0.21,24.47,80.15,61.21
# q_commonsense,56.27,44.7,79.94,81.21,5.5,79.5,0.28,14.17,75.26,66.51
# q_count,19.64,3.1,0.49,1.71,27.33,1.23,0.05,11.98,40.37,32.9
# q_action,41.03,33.45,58.1,57.27,11.9,66.72,0.43,23.06,67.11,52.51
# q_color,53.29,43.32,12.79,61.92,11.37,28.65,68.68,63.21,64.89,63.25
# q_type,31.64,26.2,84.36,53.07,0.3,20.47,4.18,27.68,39.2,27.7
# q_subcategory,41.35,34.49,36.28,37.07,2.48,41.52,0.84,20.92,53.12,45.71
# q_causal,3.91,4.21,13.55,14.01,0.46,1.52,0.0,1.27,4.72,14.31
# """

# # Using StringIO to simulate reading from a file
# csv_data = StringIO(data)

# # Read the CSV data
# df = pd.read_csv(csv_data, index_col=0)

# # Transpose the DataFrame
# transposed_df = df.transpose()

# # Set the tasks as the index (since the tasks are in the column names of the original CSV)
# transposed_df.index.name = 'task'
# transposed_df.to_csv('acc_metrics/naiveblip_seq_ft.csv')
# # Display the transposed DataFrame
# print(transposed_df)


class NaiveBLIP2(NaiveBlip2VQACL):
	def __init__(self, config):
		super().__init__(config, pool_size, prompt_pool, lambda_l2p)
		from transformers import AutoProcessor
		self.use_cap_loss = use_cap_loss
		self.num_answers = num_answers
		self.label2ans = label2ans
		self.bce_loss = nn.BCEWithLogitsLoss()
		self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
		for name, param in self.vision_model.named_parameters():
			param.requires_grad = False
		print("Freeze vision encoder")
		self.vision_model = self.vision_model.eval()
		num_layers = len(self.qformer.encoder.layer)
		# Freeze all parameters of the query transformer by default
		for param in self.qformer.parameters():
			param.requires_grad = False

		if ft_layers == 'query_tokens':
			print("Unfreeze only the query tokens")
			self.query_tokens.requires_grad = True
			self.language_projection_answers.requires_grad = True
			self.language_projection_questions.requires_grad = True
		
		print("Freeze Language model")
		for name, param in self.language_model.named_parameters():
			param.requires_grad = False
