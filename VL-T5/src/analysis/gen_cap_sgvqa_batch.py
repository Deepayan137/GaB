import os
import json
import torch
from PIL import Image
from transformers import Blip2Config, AutoProcessor, Blip2ForConditionalGeneration
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '../')
from Question_type import *

class GenCap(object):
    def __init__(self):
        self.device = 'cuda'
        model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = AutoProcessor.from_pretrained(model_name)
        config = Blip2Config.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.model.to(self.device)

    def generate_captions(self, data):
        max_new_tokens = 40
        batch_generator, total_batches = self._batchify(data, 32)
        pairs = []
        for i, batch in enumerate(tqdm(batch_generator)):
            data_subset, inputs = batch['data'], batch['inputs']
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, 
                	num_beams=5, temperature=0.9, do_sample=True, repetition_penalty=1.2)
                captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                pairs.extend(zip(data_subset, captions))
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                continue
            print(f"{i+1} out of {total_batches} completed")
        return pairs

    def _batchify(self, data, batch_size):
        num_batches = (len(data) + batch_size - 1) // batch_size
        image_dir = 'gvqa'
        def batch_generator():
            for i in range(num_batches):
                data_subset = data[i * batch_size:(i + 1) * batch_size]
                images, prompts = [], []
                for _d in data_subset:
                    img_name = f"{_d['image_id']}.jpg"
                    img_path = os.path.join(f"../datasets/{image_dir}", img_name)
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)
                    # prompts.append(prompt)
                batch_inputs = self.processor(images, return_tensors="pt", padding=True, truncation=True).to(self.device)
                yield {"data": data_subset, "inputs": batch_inputs}
        return batch_generator(), num_batches

if __name__ == "__main__":
    task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
    task = Sg_task['function']['oarlks'][task_idx]
    gencap = GenCap()
    if task_idx > -1:
        data = np.load(f'../datasets/npy/function/fcl_mmf_{task}_train.npy', allow_pickle=True)
        pairs = gencap.generate_captions(data)
        new_data = []
        for i, pair in enumerate(pairs):
            try:
                _d, caption = pair
                _d[f"caption"] = caption.strip()
                new_data.append(_d)
            except Exception as e:
                print(f"Unexpected error at index {i}: {e}")
        dest_dir = 'npy_cap_all/function'
        os.makedirs(dest_dir, exist_ok=True)
        with open(f'{dest_dir}/fcl_mmf_{task}_train.json', 'w') as json_file:
            json.dump(new_data, json_file, indent=4)
        print("Finished")
        print(f"Total samples: {len(new_data)}")
