import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from src.param import parse_args
from random import sample
import numpy as np
import random
random.seed(21)

# Function to draw bounding box on image
def draw_bounding_box(image, coordinates):
    draw = ImageDraw.Draw(image)
    draw.rectangle(coordinates, outline="red", width=10)
    return image

if __name__ == "__main__":
	args = parse_args()
	data_info_path = (f'../datasets/npy/{args.scenario}/fcl_mmf_' + f'scenetext_train.npy')
	data_info_dicts = np.load(data_info_path, allow_pickle=True)
	num_samples = len(data_info_dicts)
	indices = range(num_samples)
	rand_indices = sample(indices, 9)
	images = []
	for idx in rand_indices:
		data = data_info_dicts[idx]
		img_id = data['image_id']
		img_path = os.path.join('../datasets/textvqa_train', f'{img_id}.jpg')
		image = Image.open(img_path)
		question = data['question']
		answer = data['answer'][0]
		import pdb;pdb.set_trace()
		info = data['ocr_info']
		for item in info:
			text = item['word'].lower()
			# if text == answer:
			cords = (item['bounding_box']['top_left_x'], item['bounding_box']['top_left_y'],
				 item['bounding_box']['top_left_x'] + item['bounding_box']['width'], 
				 item['bounding_box']['top_left_y'] + item['bounding_box']['height'])
			print(f"Drawing bbox around {text} on {img_path}")
			image = draw_bounding_box(image, cords)
		images.append(image)
	# Create a 3x3 grid and plot images
	fig, axs = plt.subplots(3, 3, figsize=(15, 15))
	for ax, img in zip(axs.ravel(), images):
		ax.imshow(img)
		ax.axis('off')

	# Save the final image
	os.makedirs('viz', exist_ok=True)  # Create viz directory if it doesn't exist
	plt.savefig('viz/sce_text.jpg')
	plt.show()