import sys
sys.path.append("../")
import torch
from pathlib import Path
import json
from transformers import AutoProcessor
from PIL import Image
from vqa_model_blip import NaiveBLIP2
from transformers import T5Config, BartConfig, Blip2Config
from tqdm import tqdm
import os
import random

TASK_LIST = ['q_recognition','q_location','q_judge', 'q_commonsense', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory','q_causal']

def get_model(task=None, head_root_path=Path('/fastwork/dtalon/mm-cl/vqacl/snap/naiveblip_qtoken/'), device=torch.device('cuda')):
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_name, train=True)
    
    config = Blip2Config.from_pretrained(model_name)
    model = NaiveBLIP2.from_pretrained(model_name, config=config)

    if task is not None:
        print(f"Loading head {task}_lm_head.pth")
        # state_dict = torch.load('/fastwork/dtalon/mm-cl/vqacl/snap/naiveblip_qtoken/q_commonsense_LAST.pth')['model']
        # model.load_state_dict(state_dict)
        state_dict = torch.load(head_root_path / f"{task}_projection.pth")
         #breakpoint()
        model.language_projection.load_state_dict(state_dict)
    
    model.to(device)
    return model, processor

def inference_qa(model, processor, image, device=torch.device('cuda')):
    
    pixel_values = processor(image, max_length=20, 
                    truncation=True, return_tensors="pt")['pixel_values'].to(device)
    output = model.generate(
                        pixel_values=pixel_values, 
                        max_length=20, 
                        do_sample=True)

    pred_ans =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)
    return pred_ans

def inference(target_task, data_subset, out_file_path, img_data_path):
    target_task_idx = TASK_LIST.index(target_task)
    for idx in range(target_task_idx):
        # previous tasks
        task = TASK_LIST[idx]
        model, processor = get_model(task)
        generated_qa = {}
        num_task_replay_samples = 5000 / target_task_idx
        for _d in tqdm(data_subset):
            img_name = f"{_d['img_id']}.jpg"
            split = "train" if 'train2014' in img_name else 'val'
            img_path = os.path.join(img_data_path / f"{split}2014", img_name)
            image = Image.open(img_path).convert("RGB")
            qa = inference_qa(model, processor, image)
            gen = qa[0].strip()
            generated_qa.update({_d["img_id"]: gen})
            if len(generated_qa) >= num_task_replay_samples:
                break

        out_file = out_file_path / f'{task}_on_{target_task}.json'
        with open(out_file, 'w') as f:
            json.dump(generated_qa, f, indent=4)

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    
    current_task = 'q_action'
    data_path = f"/fastwork/dtalon/mm-cl/vqacl/datasets/vqa/Partition_Q/karpathy_train_{current_task}.json"
    out_file_path = Path(f"/fastwork/dtalon/mm-cl/vqacl/out/")
    img_data_path = Path("/fastwork/dtalon/mm-cl/vqacl/datasets/COCO/")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    data = data[:10]

    inference(current_task, data, out_file_path= out_file_path, img_data_path = img_data_path)
    


