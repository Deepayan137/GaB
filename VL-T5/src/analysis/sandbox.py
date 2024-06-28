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


import os
import json
import numpy as np
np_path=('../datasets/npy/function/fcl_mmf_object_train.npy')
data = np.load(np_path, allow_pickle=True)
new_data = []
for datum in data:
    new_data.append(datum)
# new_data=[]
# for datum in data:
#     new_datum = {}
#     for k,v in datum.items():
#         if k.startswith("Q_"):
#             new_datum['Q'] = datum[k]
#         elif k.startswith("A_"):
#             new_datum['A'] = datum[k]
#         else:
#             new_datum[k] = datum[k]
#     new_data.append(new_datum)
# import pdb;pdb.set_trace()
with open('../datasets/npy/function/fcl_mmf_object_train.json', 'w') as f:
    json.dump(new_data, f, indent=4)






