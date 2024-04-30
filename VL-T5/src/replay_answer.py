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

def get_model(ckpt_path=Path('/fastwork/dtalon/mm-cl/vqacl/snap/naiveblip_cl_qtoken/q_count_LAST.pth'), device=torch.device('cuda')):
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_name, train=True)
    
    config = Blip2Config.from_pretrained(model_name)
    model = NaiveBLIP2.from_pretrained(model_name, config=config)

    checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(state_dict, strict=False)
    model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
    model.language_projection.load_state_dict(checkpoint['model']['language_projection'])
    # model.language_projection.load_state_dict(state_dict['language_projection'])
    # breakpoint()
    # model.query_tokens
    
    model.to(device)
    return model, processor

def inference_qa(model, processor, image, question, device=torch.device('cuda')):
    sent = f"Question: {question.lower()} Answer:"
    print(sent)
    in_data= processor(image, text=sent, max_length=20, 
                    truncation=True, return_tensors="pt").to(device)
    input_ids = in_data['input_ids']
    pixel_values = in_data['pixel_values']
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long().to(device)
    max_new_tokens=2
    output = model.generate(input_ids=input_ids,pixel_values=pixel_values,attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,repetition_penalty=1.2)
    pred_ans =  processor.tokenizer.batch_decode(output, skip_special_tokens=True)

    return pred_ans

def inference(data_subset, img_data_path):

        model, processor = get_model()
        generated_qa = {}
        for _d in tqdm(list(data_subset.keys())):
            img_name = f"{_d}.jpg"
            split = "train" if 'train2014' in img_name else 'val'
            img_path = os.path.join(img_data_path / f"{split}2014", img_name)
            image = Image.open(img_path).convert("RGB")
            img_qa = {}
            
            for q_idx, it in enumerate(data_subset[_d].items()):
                ans = inference_qa(model, processor, image, it[1])[0]
                ans = ans.strip()
                img_qa.update({it[0]: {'Q': it[1], 'A':ans}})
            generated_qa.update({_d: img_qa})
        
        return generated_qa

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    
    question_file_path = Path(f"/fastwork/dtalon/mm-cl/vqacl/snap/naiveblip_qtoken/q_action_replay.json")
    img_data_path = Path("/fastwork/dtalon/mm-cl/vqacl/datasets/COCO/")

    out_file = Path('/fastwork/dtalon/mm-cl/vqacl/out') / f'self_qa_q_action.json'
    with open(question_file_path, 'r') as f:
        data = json.load(f)
    
    # data = {it[0]: it[1] for idx, it in enumerate(data.items()) if idx < 20}
    question_answers = inference(data, img_data_path = img_data_path)
    
    with open(out_file, 'w') as f:
        json.dump(question_answers, f, indent=4)