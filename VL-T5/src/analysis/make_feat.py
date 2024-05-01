import torch
import torch.nn as nn
from PIL import Image
from transformers import T5Config, BartConfig, Blip2Config

from torch.utils.data import DataLoader, Dataset, Subset
from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
from src.param import parse_args
# from src.vqacl import Trainer
import sys
from tqdm import *
from Question_type import All_task, Category_splits
import os
from transformers import AutoProcessor
from src.vqa_model_blip import NaiveBLIP2
backbone = "Salesforce/blip2-opt-2.7b"
config = Blip2Config.from_pretrained(backbone)
processor = AutoProcessor.from_pretrained(backbone)
model = NaiveBLIP2.from_pretrained(backbone, config=config)
device = 'cuda'
model = model.to(device)
split = 'train'
coco_Ours = All_task
train_dset = VQADataset(f"karpathy_{split}", True)
val_dset = VQADataset(f"karpathy_val", True)
args = parse_args()
args.backbone = backbone

os.makedirs(f'../datasets/COCO/train2014_feat', exist_ok=True)
os.makedirs(f'../datasets/COCO/val2014_feat', exist_ok=True)

def write_to_disk(id_feat_pairs):
	for pair in id_feat_pairs:
		img_id, query_outputs, vision_outputs = pair
		query_hidden_states = query_outputs['query_hidden_states']
		query_pooler_output = query_outputs['query_pooler_output']

		vision_hidden_states = vision_outputs['vision_hidden_states']
		vision_pooler_output = vision_outputs['vision_pooler_output']
		split = img_id.split('_')[1]
		dest = f'../datasets/COCO/{split}_feat'
		savepath = os.path.join(dest, img_id + '.pth')
		features = {
			'vision_outputs':(vision_hidden_states, vision_pooler_output), 
			'query_outputs':(query_hidden_states, query_pooler_output)}
		torch.save(features, savepath)

class COCODataset(Dataset):
	def __init__(self, root):
		files = os.listdir(root)
		self.image_paths = [os.path.join(root, file) for file in files if file.endswith('.jpg')]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		fname = os.path.basename(image_path)
		image_id = fname.split('.')[0]
		image = Image.open(image_path).convert("RGB")
		
		inputs = processor(image, return_tensors="pt")
		
		return {"pixel_values":inputs["pixel_values"], "img_id":image_id}

def get_dataset_subset(dataset, num_splits, split_index):
    # Calculate split size and starting index for the subset
    split_size = len(dataset) // num_splits
    start_index = split_index * split_size
    end_index = start_index + split_size if split_index != num_splits - 1 else len(dataset)
    return Subset(dataset, range(start_index, end_index))

# Get SLURM_ARRAY_TASK_ID or default to 0 (useful for testing without Slurm)
task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
num_splits = 10  # Example: 10-way split, adjust based on your --array option in Slurm

# Initialize datasets
train_dataset_full = COCODataset("../datasets/COCO/train2014")
train_dataset = get_dataset_subset(train_dataset_full, num_splits, task_id)
train_loader = DataLoader(train_dataset, batch_size=80)

val_dataset_full = COCODataset("../datasets/COCO/val2014")
val_dataset = get_dataset_subset(val_dataset_full, num_splits, task_id)
val_loader = DataLoader(val_dataset, batch_size=80)


for loader in [train_loader, val_loader]:
    with tqdm(total=len(loader), unit='batch', desc=f"Processing {'train' if loader == train_loader else 'val'} dataset") as pbar:
        for batch in loader:

            image_ids = batch['img_id']
            pixel_values = batch['pixel_values'].squeeze().to(device)  # Ensure pixel_values is correctly shaped
            
            outputs = model.get_features(pixel_values)
            
            query_outputs = outputs["query_outputs"]
            vision_outputs = outputs["vision_outputs"]
            
            # Extract hidden states and pooler outputs
            query_hidden_states = query_outputs.last_hidden_state.to('cpu')  # Shape: [batch_size, seq_len, hidden_size]
            query_pooler_output = query_outputs.pooler_output.to('cpu') if query_outputs.pooler_output is not None else None
            
            vision_hidden_states = vision_outputs.last_hidden_state.to('cpu')  # Shape: [batch_size, seq_len, hidden_size]
            vision_pooler_output = vision_outputs.pooler_output.to('cpu') if vision_outputs.pooler_output is not None else None
            
            id_feat_triplets = [
                (img_id, 
                 {'query_hidden_states': query_hidden_states[i], 'query_pooler_output': query_pooler_output[i]},
                 {'vision_hidden_states': vision_hidden_states[i], 'vision_pooler_output': vision_pooler_output[i]}
                ) for i, img_id in enumerate(image_ids)
            ]
            write_to_disk(id_feat_triplets)
            pbar.update(1)

