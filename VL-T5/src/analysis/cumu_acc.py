import os
import json

import pandas as pd

All_task = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count']
# All_task = ['q_location']
class Analysis(object):
	def __init__(self, paths):
		paths_with_pred = self._check_pred_dir_exists(paths)
		paths_with_causal = self._check_if_causal_exists(paths_with_pred)
		print(len(paths_with_causal))
		self.get_matrix(paths_with_causal)

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
			if os.path.exists(os.path.join(path[0], "predictions", "q_causal_acc.json"))]
		return with_causal

	def parse_acc_json(self, json_path):
		task = '_'.join(os.path.basename(json_path).split('_')[:2])
		with open(json_path, 'r') as f:
			data = json.load(f)
		acc_dict = {}
		for task_ in All_task:
			acc_dict[task_]= data[task][task_][0]['Test/overall']
		return acc_dict

	def get_matrix(self, paths):
		for path in paths:
			conf_dict = {}
			for task in All_task:
				json_path = os.path.join(path, f"{task}_acc.json")
				conf_dict[task]=self.parse_acc_json(json_path)
			model_name = os.path.dirname(path).split('/')[-1]
			target_path = os.path.join('metrics', f'{model_name}.csv')
			df = pd.DataFrame.from_dict(conf_dict, orient='index')
			df.to_csv(target_path)
			print(f"csv saved @ {target_path}")
		print("Done")

if __name__ == "__main__":
	savepath = "/leonardo_scratch/fast/IscrC_CLRT-VLM/VQACL/VL-T5/snap"
	model_names = ['naiveblip_cl_syn_filtered']
	# model_names = os.listdir(savepath)
	f = lambda x: os.path.join(savepath, x)
	model_paths = list(map(f, model_names))
	Analysis(model_paths)

