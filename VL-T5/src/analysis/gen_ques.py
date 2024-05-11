import json
import os
import spacy
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2Config
from src.vqa_model_blip import NaiveBLIP2
from src.param import parse_args

nlp = spacy.load("en_core_web_sm")
args = parse_args()

class GenQues():
    def __init__(self, ckpt_path):
        model_name = "Salesforce/blip2-opt-2.7b"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_model(model_name, ckpt_path)

    def get_model(self, model_name, ckpt_path):
        self.processor = AutoProcessor.from_pretrained(model_name)
        config = Blip2Config.from_pretrained(model_name)
        self.model = NaiveBLIP2.from_pretrained(model_name, config=config)
        self.model.to(self.device)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.query_tokens.data.copy_(checkpoint['model']['query_tokens'])
            self.model.language_projection_questions.load_state_dict(checkpoint['model']['language_projection_questions'])

    def inference_qa(self, image_path, max_new_tokens=20):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, truncation=True, return_tensors="pt").to(self.device)
        batch = {'pixel_values': inputs["pixel_values"]}
        output = self.model.get_questions(batch, max_new_tokens=max_new_tokens, num_beams=5, num_return_sequences=3)
        # pred_question = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output

if __name__ == "__main__":
    path = "../datasets/vqa/Partition_Q_V2/"
    fname = "karpathy_train_q_judge.json"
    source = os.path.join(path, fname)
    dest_root = "/desired/destination/directory"  # Make sure to define this
    dest = os.path.join(dest_root, fname)
    ckpt_path = 'snap/naiveblip_cl_syn_st/q_location_LAST.pth'
    gen_ques = GenQues(ckpt_path)
    
    with open(source, 'r') as f:
        data_subset = json.load(f)
    
    for _d in tqdm(data_subset):
        img_name = f"{_d['img_id']}.jpg"
        split = "train" if 'train2014' in img_name else 'val'
        img_path = os.path.join(f"../datasets/COCO/{split}2014", img_name)
        questions = gen_ques.inference_qa(img_path)
        print(questions)
        import pdb;pdb.set_trace()