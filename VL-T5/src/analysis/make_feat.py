import torch
import torch.nn as nn
from PIL import Image
from transformers import T5Config, BartConfig, Blip2Config
from torch.utils.data import DataLoader, Dataset
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
		if os.path.exists(image_path):
		    image = Image.open(image_path).convert("RGB")
		else:
		    raise f"{image_path} does not exists"
		inputs = processor(image, return_tensors="pt")
		
		return {"pixel_values":inputs["pixel_values"], "img_id":image_id}

train_dataset = COCODataset("../datasets/COCO/train2014")
train_loader = DataLoader(train_dataset, batch_size=32)
val_dataset = COCODataset("../datasets/COCO/val2014")
val_loader = DataLoader(val_dataset, batch_size=32)

os.makedirs(f'../datasets/COCO/train2014_feat', exist_ok=True)
os.makedirs(f'../datasets/COCO/val2014_feat', exist_ok=True)


def write_to_disk(id_feat_pairs):
	for pair in id_feat_pairs:
		img_id, feat = pair 
		split = img_id.split('_')[1]
		
		dest = f'../datasets/COCO/{split}_feat'
		savepath = os.path.join(dest, img_id + '.pth')
		torch.save(feat, savepath)


for loader in [train_loader, val_loader]:
	with tqdm(total=len(loader),unit='batch') as pbar:
	    for batch in tqdm(loader):
	        image_ids = batch['img_id']
	        pixel_values = batch['pixel_values'].squeeze().to(device) # bs, 36, 2048
	        query_features = model.get_features(pixel_values)
	        id_feat_pairs = list(zip(image_ids, query_features.to('cpu')))
	        
	        write_to_disk(id_feat_pairs)
	        # pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss:.4f}')
	        pbar.update(1)
