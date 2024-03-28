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


# class NaiveInstructBLIP2(InstructBlipForConditionalGeneration):
#     def __init__(self, config, num_answers=None, label2ans=None, ft_layers='query_tokens'):
#         super().__init__(config)
#         from transformers import AutoProcessor
#         self.num_answers = num_answers
#         self.label2ans = label2ans
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.processor = AutoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
#         self.processor.tokenizer.padding_side = 'right'
#         self.processor.tokenizer.truncation_side = 'right'
#         for name, param in self.vision_model.named_parameters():
#             param.requires_grad = False
#         print("Freeze vision encoder")
#         self.vision_model = self.vision_model.eval()
#         self.query_tokens.requires_grad = True
#         # for param in self.qformer.parameters():
#         #     param.requires_grad = False

#         print("Freeze Language model")
#         for name, param in self.language_model.named_parameters():
#             param.requires_grad = False


#     @torch.no_grad()
#     def test_step(self, batch, task, **kwargs):
#         self.eval()
#         device = next(self.parameters()).device
#         pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
#         input_ids = batch['input_ids'].to(device) # bs, 20
#         lm_labels = batch["target_ids"].to(device) #[bs, 5]
#         qformer_input_ids = batch['qformer_input_ids'].to(device)
#         max_new_tokens = 5
#         output = self.generate(input_ids=input_ids,
#             pixel_values=pixel_values,
#             qformer_input_ids=qformer_input_ids,
#             # max_new_tokens=max_new_tokens,
#             eos_token_id=50118,
#             repetition_penalty=1.2)
#         result = {}
#         # output[output == -1] = self.processor.tokenizer.pad_token_id
#         result['token_ids'] = output
#         result['pred_ans'] = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True).strip('\n')
#         return result

#     def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False):
#         device = next(self.parameters()).device
#         pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
#         input_ids = batch['input_ids'].to(device) # bs, 20
#         lm_labels = batch["target_ids"].to(device) #[bs, 5]
#         attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
#         qformer_input_ids = batch['qformer_input_ids'].to(device)
#         output = self(input_ids=input_ids,
#             pixel_values=pixel_values,
#             qformer_input_ids=qformer_input_ids,
#             labels=lm_labels)
#         assert 'loss' in output
#         B, L = lm_labels.size()
#         loss = output['loss'] # 400 (bs*5)
#         result = {
#             'loss': loss
#         }
#         result['logits'] = output['logits']
#         result['BL'] = (B, L)
#         if 'loss_memory' in output:
#             result['loss_memory'] = output['loss_memory']  #(output['loss_memory_Q'], output['loss_memory_V'])
#         if 'loss_memory_new' in output:
#             result['loss_memory_new'] = output['loss_memory_new']
#         return result

