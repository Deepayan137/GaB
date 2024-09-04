import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

blip_base = {
	
	'q_recognition':32.21,
	'q_location':16.78,
	'q_judge':61.6,
	'q_commonsense':66.03,
	'q_count':30.36,
}
def plot_conf_mat(file_path, destination):
	# Load your CSV file
	fname = os.path.basename(file_path)
	fname = fname.split('.')[0]+'.png'
	target_file_path = os.path.join(destination, fname)
	
	df = pd.read_csv(file_path, index_col=0)
	# Plotting the confusion matrix
	plt.figure(figsize=(10, 8))  # Adjust the size of the plot as needed
	sns.heatmap(df, annot=True, cmap='Blues', fmt='g')  # 'g' for generic number formatting
	plt.title('Confusion Matrix')
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.savefig(target_file_path)
	plt.show()
	
	plt.close()

if __name__ == "__main__":
	savepath = "acc_metrics"
	destination = os.path.join("acc_metrics", "conf_matrix")
	os.makedirs(destination, exist_ok=True)
	file_names = os.listdir(savepath)
	f = lambda x: os.path.join(savepath, x)
	file_paths = list(map(f, file_names))
	for file_path in file_paths:
		try:
			plot_conf_mat(file_path, destination)
		except:
			print(file_path)
