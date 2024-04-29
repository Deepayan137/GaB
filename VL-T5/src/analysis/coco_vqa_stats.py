import os
import json
import ijson
import sys
sys.path.insert(0, '../')
from Question_type import *
from tqdm import *
root='../datasets/vqa'
tr_cap_fname = os.path.join(root, 'annotations/', 'captions_train2014.json')
val_cap_fname = os.path.join(root, 'annotations/', 'captions_val2014.json')
tr_inst_fname=os.path.join(root, 'annotations/', 'instances_train2014.json')
val_inst_fname=os.path.join(root, 'annotations/', 'instances_train2014.json')
tr_vqa_fname=os.path.join(root, 'v2_mscoco_train2014_annotations.json')
val_vqa_fname=os.path.join(root, 'v2_mscoco_val2014_annotations.json')

def get_cap(fname):
	with open(fname, 'r') as f:
		cap_ann_dict = json.load(f)
	ImgId_cap_map={}
	for cap_data in cap_ann_dict['annotations']:
		image_id = cap_data['image_id']
		caption = cap_data['caption']
		if  image_id not in ImgId_cap_map:
			ImgId_cap_map[image_id] = []
		ImgId_cap_map[image_id].append(caption)
	return ImgId_cap_map

def vqa(id_cap_tr, id_cap_val, label_dict):
	dest_root = '../datasets/vqa/Partition_Q_V2'
	qtype_dict = {}
	qtype_ex = {}
	for task_idx, task in enumerate(tqdm(All_task)):
		print(f"Now is task {task}")
		qtype_dict[task] = {}
		qtype_ex[task] = {}
		data_info_path = ('../datasets/vqa/Partition_Q_V2/karpathy_train_' + f'{All_task[task_idx]}.json')
		new_data_info_path = ('../datasets/vqa/Partition_Q_V2_meta/karpathy_train_' + f'{All_task[task_idx]}.json')
		with open(data_info_path, 'r') as f:
			data_info_dicts = json.load(f)
		new_data_info_dicts = []
		for _d in data_info_dicts:
			image_name = _d['img_id']
			split, image_id = image_name.split('_')[1], int(image_name.split('_')[2])
			if split == 'train2014':
				caption = id_cap_tr[image_id]
			else:
				caption = id_cap_val[image_id]
			if image_name in label_dict:
				cat_name = label_dict[image_name]
			else:
				cat_name = 'None'
			_d['caption'] = caption
			_d['category'] = cat_name
			new_data_info_dicts.append(_d)
		with open(new_data_info_path, 'w') as f:
			json.dump(new_data_info_dicts, f, indent=4)


def get_labels(fname):
	fd = open(fname,'r')
	objs = ijson.items(fd, 'categories.item')
	labels = (o for o in objs)
	count = 0
	label_dict={}
	for label in labels:
		# print('id:{}, category:{}, super category:{}'.format(label['id'], label['name'], label['supercategory']))
		# count += 1
		if label['name'] not in label_dict:
			label_dict[label['id']] = label['name']
	fd.close()
	with open(os.path.join('../datasets', 'ImgId_cate_map.json')) as fp:
		ImgId_cate_map = json.load(fp)
	ImgId_label_map={}
	
	for img_id, label_id in ImgId_cate_map.items():
		ImgId_label_map[img_id]=label_dict[label_id]
	return ImgId_label_map

if __name__ == "__main__":
	label_dict = get_labels(tr_inst_fname)
	id_cap_tr = get_cap(tr_cap_fname)
	id_cap_val = get_cap(val_cap_fname)
	vqa(id_cap_tr, id_cap_val, label_dict)
	# import pdb;pdb.set_trace()
	# vqa=get_vqa(tr_vqa_fname)
	
		
		

