import os
import json
import random
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import *

All_task = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory','q_causal']

random.seed(42)

class Visualize(object):
	def __init__(self, paths):
		paths_with_pred = self._check_pred_dir_exists(paths)
		paths_with_causal = self._check_if_causal_exists(paths_with_pred)
		print(len(paths_with_causal))
		self._main(paths_with_causal)

	def _check_pred_dir_exists(self, paths):
		# Create a list of tuples (path, has_predictions)
		path_info = [(path, os.path.isdir(os.path.join(path, 'predictions'))) for path in paths]

		# Filter paths based on whether they have 'predictions'
		paths_with_predictions = list(filter(lambda x: x[1], path_info))
		paths_without_predictions = list(filter(lambda x: not x[1], path_info))
		print(f"""Total:{len(paths)}. With pred dir:{len(paths_with_predictions)}. 
			Without pred dir:{len(paths_without_predictions)}.""")
		print(f"Paths without pred dir:{paths_without_predictions}")
		return paths_with_predictions

	def _check_if_causal_exists(self, paths):
		with_causal = [os.path.join(path[0], "predictions") for path in paths 
			if os.path.exists(os.path.join(path[0], "predictions", "q_causal_LAST_gt_pred.json"))]
		return with_causal

	def parse_pred_json(self, json_path):
		task = '_'.join(os.path.basename(json_path).split('_')[:2])
		with open(json_path, 'r') as f:
			data = json.load(f)
		with open('snap/blip_base/predictions/base_gt_pred.json', 'r') as f:
			base_data = json.load(f)
		pred_dict = {}
		for task_ in All_task:
			pred_dict[task_]={}
			preds = data[task][task_][0]
			qids = list(preds.keys())
			sampled_qids = random.sample(qids, k=9)
			for qid in sampled_qids:
				pred_dict[task_][qid]= {
					'img_id':data[task][task_][0][qid][0],
					'question':data[task][task_][0][qid][1],
					'label':data[task][task_][0][qid][3],
					'prediction':data[task][task_][0][qid][2],
					'base_prediction':base_data['q_causal'][task_][0][qid][2]
			}
		return pred_dict

	def plot_(self, data_all_tasks, dest_name, task):
		# Create a 3x3 grid
		for k, data in enumerate(data_all_tasks):
			os.makedirs(os.path.join(dest_name, task), exist_ok=True)
			fname = os.path.join(dest_name, task, f'{All_task[k]}.jpg')
			data = list(data.values())
			fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
			for i, item in enumerate(data):

				row = i // 3
				col = i % 3

				# Load and display image
				img_path = os.path.join("/home/deepayan.das/projects/VQACL/datasets/COCO", 
					"val2014", f'{item["img_id"]}.jpg')
				img = Image.open(img_path)
				axs[row, col].imshow(img)
				axs[row, col].axis('off')  # Turn off axis

				# Display question and answers
				axs[row, col].set_title(f"""Q: {item['question']}\nGT: {item['label']}\nPred: {item['prediction']}\nBase Pred:{item['base_prediction']}""")

			plt.tight_layout()
			plt.show()
			plt.savefig(fname)
			plt.close()
	def _main(self, paths):
		for path in tqdm(paths):
			conf_dict = {}
			for task in All_task:
				dir_name = path.split('/')[-2]
				root = '/'.join(path.split('/')[:-1])
				dest_name = os.path.join('viz', dir_name)
				if not os.path.exists(dest_name):
					os.makedirs(dest_name, exist_ok=True)
				json_path = os.path.join(path, f"{task}_LAST_gt_pred.json")
				sampled_dict = self.parse_pred_json(json_path)
				preds_all_tasks = list(sampled_dict.values())
				self.plot_(preds_all_tasks, dest_name, task)


if __name__ == "__main__":
	savepath = "/home/deepayan.das/projects/VQACL/VL-T5/snap"
	model_names = os.listdir(savepath)
	f = lambda x: os.path.join(savepath, x)
	model_paths = list(map(f, model_names))
	Visualize(model_paths)

