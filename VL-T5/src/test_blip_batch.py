import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import requests
from PIL import Image

base_model_id = "Salesforce/blip2-opt-2.7b"
model = Blip2ForConditionalGeneration.from_pretrained(base_model_id)
processor = Blip2Processor.from_pretrained(base_model_id)
device = 'cuda'
model.to(device)
# from provided huggingface example -> https://huggingface.co/Salesforce/blip2-opt-2.7b
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question1 = "which dog?"
question2 = "is there a dog here?"
prompt1 = f"Question: {question1} Answer:"
prompt2 = f"Question: {question2} Answer:"
# batching
inputs = processor([raw_image,raw_image], [prompt1, prompt2], padding = True, return_tensors="pt").to(device)

# both separate 
i0 = processor(raw_image, text=prompt1, return_tensors="pt").to(device)
i1 = processor(raw_image, text=prompt2, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(**inputs)
    o0 = model.generate(**i0)
    o1 = model.generate(**i1)
    
print(processor.batch_decode(out, skip_special_tokens = True))
print(processor.batch_decode(o0, skip_special_tokens = True))
print(processor.batch_decode(o1, skip_special_tokens = True))
# logits = out['logits']
# l0 = o0['logits']
# l1 = o1['logits']

# # outputs False in both cases 
# print(torch.allclose(logits[0][:l0.shape[1]].unsqueeze(0), l0))
# print(torch.allclose(logits[1][:l1.shape[1]].unsqueeze(0), l1))
