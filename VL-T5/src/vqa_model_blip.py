from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from src.modeling_blip import Blip2VQACL
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration

class NaiveBLIP2(Blip2ForConditionalGeneration):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        from transformers import AutoProcessor
        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        for name, param in self.vision_model.named_parameters():
            param.requires_grad = False
        
        self.vision_model = self.vision_model.eval()
        # print("Freezing qformer")
        # for name, param in self.qformer.named_parameters():
        #     param.requires_grad = False
        
        print("Freezing qformer all but last layer")
        num_layers = len(self.qformer.encoder.layer)

        # for i, layer in enumerate(self.qformer.encoder.layer):
        #     # Freeze all layers except the last one
        #     if i < num_layers - 1:
        #         for param in layer.parameters():
        #             param.requires_grad = False

        self.query_tokens.requires_grad = True
        print("freeze vision encoder")
        
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False
        # self.eos_token_id = self.processor.tokenizer('\n', add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def test_step(self, batch, task, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]
        
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
        # cate_labels = batch['cate_labels'].to(device)
        # ques_labels = batch['ques_labels'].to(device)
        max_new_tokens = 2
        if task in ['q_recognition', 'q_type']:
            max_new_tokens = 3
        output = self.generate(input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2)
        result = {}
        result['token_ids'] = output
        result['pred_ans'] = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        return result

    def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False):
        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
        output = self(input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=lm_labels
            )
        assert 'loss' in output
        B, L = lm_labels.size()
        loss = output['loss'] # 400 (bs*5)
        result = {
            'loss': loss
        }
        result['logits'] = output['logits']
        result['BL'] = (B, L)
        if 'loss_memory' in output:
            result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
        if 'loss_memory_new' in output:
            result['loss_memory_new'] = output['loss_memory_new']
        return result

class BLIP2VQA(Blip2VQACL):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False):

        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]

        cate_labels = batch['cate_labels'].to(device)
        ques_labels = batch['ques_labels'].to(device)
        attention_mask = (input_ids != 1).long().to(device)
        output = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=lm_labels,
            cate_labels=cate_labels,
            ques_labels=ques_labels,
            proto_update=True,
            memory=memory,
            current_task_id=current_task_id,
            mem_num_Q = mem_num_Q,
            total_num_Q = total_num_Q,
            proto_alpha=proto_alpha,
            proto_beta=proto_beta,
            return_dict=True
        )
        assert 'loss' in output
        lm_mask = (lm_labels != -100).float()
        lm_mask = lm_mask[:, :-1]  
        B, L = lm_labels.size()
        loss = output['loss'] # 400 (bs*5)
        loss = loss.view(B, L-1) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        loss = loss * batch['scores'].to(device=device) # batch['score']: bs
        loss = loss.mean()
        result = {
            'loss': loss
        }
        result['logits'] = output['logits']
        result['BL'] = (B, L)
        # result['encoder_attention_mask'] = output['encoder_attention_mask']
        if 'loss_memory' in output:
            result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
        if 'loss_memory_new' in output:
            result['loss_memory_new'] = output['loss_memory_new']
        return result


    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]
        attention_mask = (input_ids != 1).long().to(device)

        cate_labels = batch['cate_labels'].to(device)
        ques_labels = batch['ques_labels'].to(device)
        output = self.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=10,

            **kwargs,
        )
        # import pdb;pdb.set_trace()
        result = {}
        result['token_ids'] = output
        # result['pred_ans'] = generated_sents

        return result

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from transformers import Blip2Config, AutoProcessor
    from torch.utils.data import DataLoader, Dataset
    from src.vqa_data_blip import VQADataset, VQAFineTuneDataset, FilteredVQAFineTuneDataset
    from src.param import parse_args
    from src.vqacl import Trainer
    import sys
    from tqdm import *
    from Question_type import All_task, Category_splits
    import os
    backbone = "Salesforce/blip2-opt-2.7b"
    config = Blip2Config.from_pretrained(backbone)
    processor = AutoProcessor.from_pretrained(backbone)
    # model = Blip2ForConditionalGeneration.from_pretrained(backbone, config=config)
    model = NaiveBLIP2.from_pretrained(backbone, config=config)
    task = 'q_location'
    save_path = f'snap/test/{task}_LAST.pth'
    device = 'cuda'
    model = model.to(device)
    # if os.path.exists(save_path):
    #     print(f'Loading model at {save_path}')
    #     ckpt = torch.load(save_path)
    #     model.load_state_dict(ckpt)
    split = 'train'
    coco_Ours = All_task
    train_dset = VQADataset(f"karpathy_{split}", True)
    val_dset = VQADataset(f"karpathy_val", True)
    args = parse_args()
    args.backbone = backbone

    dataset = VQAFineTuneDataset(
                coco_Ours,
                [],
                'karpathy_train',
                raw_dataset=train_dset,
                rank=0,
                topk=-1,
                verbose=True,
                args=args,
                mode='train',
                task=task,
                cates=Category_splits['G1']
            )
    dataset = FilteredVQAFineTuneDataset(dataset)
    train_loader_cate = DataLoader(
                dataset, batch_size=80, shuffle=True,
                num_workers=4, pin_memory=True, sampler=None,
                collate_fn=dataset.collate_fn)
    dataset = VQAFineTuneDataset(
                coco_Ours,
                [],
                f'karpathy_test',
                raw_dataset=val_dset,
                rank=0,
                topk=-1,
                verbose=True,
                args=args,
                mode='val',
                task=task,
                cates=[i for i in range(80)]
            )
    dataset = FilteredVQAFineTuneDataset(dataset)
    val_loader_cate =  DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=4, pin_memory=True, sampler=None,
                collate_fn=dataset.collate_fn, drop_last=False)
    epoch_results = {
                        'loss': 0.,
                    }
    # config.use_decoder_only_language_model = False
    # for name, param in model.vision_model.named_parameters():
    #     param.requires_grad = False
    # model.vision_model = model.vision_model.eval()
    # print("freeze vision encoder")
    # for name, param in model.language_model.named_parameters():
    #     param.requires_grad = False
    eos_token_id = processor.tokenizer('\n', add_special_tokens=False).input_ids[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    num_epochs = 3
    trainer = Trainer(args, All_task)
    def preprocess(text):
        # Convert to lowercase, strip whitespace, remove punctuation, etc.
        text = text.lower().strip()
        return text
    
    def evaluate(predictions, truths):
        total = len(predictions)
        correct = 0

        for pred, truth in zip(predictions, truths):
            if preprocess(pred) in preprocess(truth) or preprocess(truth) in preprocess(pred):
                correct += 1
        accuracy = correct / total
        return accuracy

    def validation(model, loader):
        # preds = []
        # truths = []
        # print("Validating")
        # for batch in tqdm(loader):
        #     pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        #     input_ids = batch['input_ids'].to(device) # bs, 20
        #     lm_labels = batch["target_ids"].to(device)
        #     attention_mask = (input_ids != processor.tokenizer.pad_token_id).long().to(device)
        #     output = model.generate(input_ids=input_ids,
        #         pixel_values=pixel_values,
        #         attention_mask=attention_mask, max_new_tokens=20)
        #     out_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        #     preds.append(out_text[0])
        #     truths.append(batch['answers'][0])
        acc = trainer.evaluate(loader)
        print(f"Accuracy:{acc}")
    
    # validation(model, val_loader_cate)
    print("Training starts:")
    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader_cate), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch in tqdm(train_loader_cate):
                pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
                input_ids = batch['input_ids'].to(device) # bs, 20
                lm_labels = batch["target_ids"].to(device) 
                attention_mask = (input_ids != processor.tokenizer.pad_token_id).long().to(device)
                output = model(input_ids=input_ids,pixel_values=pixel_values,attention_mask=attention_mask, 
                    labels=lm_labels)
                loss = output.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss:.4f}')
                pbar.update(1)
            validation(model, val_loader_cate)
    torch.save(model.state_dict(), os.path.join('snap/test/q_location.pth'))
