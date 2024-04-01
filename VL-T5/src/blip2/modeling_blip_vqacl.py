import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGeneration, Blip2ForConditionalGenerationModelOutput)

class Blip2VQACL(Blip2ForConditionalGeneration):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.Q_task_mem_proto = {}
        self.V_task_mem_proto = {}
        self.Q_task_cur_proto = {}
        self.V_task_cur_proto = {}
        self.Q_prototype_num = {}
        self.V_prototype_num = {}
        print("Q_task_mem_proto and Q_task_cur_proto")
        # if freeze_vit:
        for name, param in self.vision_model.named_parameters():
            param.requires_grad = False
        self.vision_model = self.vision_model.eval()
        for name, param in self.qformer.named_parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = True
        print("freeze vision encoder")
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False
        
    
    def cosine_similarity_multi(self, a, b, labels=None, rep="real"):
        """
        Compute the cosine similarity between two vectors

        Parameters:
        ----------
        a:  Tensor(N_a,D) prototype N_a type of question (10, 768) or category (80, 768)
        b:  Tensor(N_b,D) average question embeddings in a given batch (80, 768) 
        rep: str
            Representation to compute cosine similarity: real | bipolar | tanh
        Return
        ------
        similarity: Tensor(N_a,N_b) (10 or 80, 80)
        """
        sim_act = nn.Tanh()  # meta-train: tanh
        a_normalized = F.normalize(sim_act(a), dim=1) # [class_num, 768]
        b_normalized = F.normalize(sim_act(b), dim=1) #[bs, dim]
        similiarity = F.linear(a_normalized, b_normalized).transpose(1,0) # [bs, class_num]
        max_idx = torch.argmax(similiarity, dim=1) #[bs] 
        selected_prototype = a[max_idx] # [bs, 768]

        if labels is not None:
            labels = torch.topk(labels, 1)[1].squeeze(1) # convert one-hot to label
            acc = (max_idx == labels).sum()//labels.shape[0]
            # print("current retreieval acc is:", 100*acc,'%')
        else:
            acc = -1

        return selected_prototype, max_idx, acc


    def update_prototype(self, current_Q_prototype, 
        current_V_prototype, current_num_Q, 
        current_num_V, current_task_id, 
        proto_alpha, proto_beta):

        if current_task_id not in self.Q_task_cur_proto:
            self.Q_task_cur_proto[current_task_id] = current_Q_prototype
            self.Q_prototype_num = current_num_Q
            self.V_prototype_num = current_num_V
            self.V_prototype = current_V_prototype
            if current_task_id == 0:
                self.Q_prototype = current_Q_prototype
            else:
                self.Q_prototype[current_task_id] = current_Q_prototype[current_task_id]
        else:
            self.Q_task_cur_proto[current_task_id] = current_Q_prototype
            if current_task_id != 0:
                if current_task_id not in self.Q_task_mem_proto:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0
                    self.Q_task_mem_proto[current_task_id] = current_Q_prototype_mem
                else:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0
                    self.Q_task_mem_proto[current_task_id] = proto_alpha*self.Q_task_mem_proto[current_task_id] + (1-proto_alpha)*current_Q_prototype_mem.detach()

                self.Q_prototype = self.Q_task_mem_proto[current_task_id].detach()
                self.Q_prototype[current_task_id] = self.Q_task_cur_proto[current_task_id][current_task_id].detach()
            else:
                self.Q_prototype = self.Q_task_cur_proto[current_task_id]


            self.V_prototype = proto_beta * self.V_prototype + (1-proto_beta) * current_V_prototype
            self.Q_prototype_num = self.Q_prototype_num.detach() + current_num_Q
            self.V_prototype_num = self.V_prototype_num.detach() + current_num_V

    def calculate_current_prototype(self, fc_hidden_Q, labels):
        
        fc_hidden_Q = torch.mean(fc_hidden_Q, dim=1)  # ---- mean-pooling

        div_item_ = torch.sum(labels, dim=0).unsqueeze(1).repeat(1, 2560)  # [num_classes1, dim]
        ones = torch.ones((labels.shape[1], fc_hidden_Q.shape[-1])).to(fc_hidden_Q.device)  # [num_classes1, dim]
        div_item = torch.where(div_item_ <= 0, ones, div_item_)

        current_prototype_Q = torch.matmul(torch.transpose(labels, 0, 1),
                                           fc_hidden_Q) / div_item  # [num_classes1, dim]

        current_num = torch.sum(labels, dim=0)
        return current_prototype_Q, current_num
    
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        **kwargs,
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # ========================================================
        if 'cate_labels' in kwargs:
            cate_labels = kwargs['cate_labels']  # [bs, num_classes]
        if 'ques_labels' in kwargs:
            ques_labels = kwargs['ques_labels']  # [bs, num_classes]

        if 'proto_alpha' in kwargs:
            proto_alpha = kwargs['proto_alpha']
        if 'proto_beta' in kwargs:
            proto_beta = kwargs['proto_beta']

        if 'current_task_id' in kwargs:
            current_task_id = kwargs['current_task_id']
    
        if 'proto_update' in kwargs and kwargs['proto_update']:  # only for training
            current_prototype_Q, current_num_Q = self.calculate_current_prototype(input_embeds.to(language_model_inputs.device),
                                                                                  ques_labels)
            current_prototype_V, current_num_V = self.calculate_current_prototype(language_model_inputs,
                                                                                  cate_labels)
            # import pdb;pdb.set_trace()
            if 'memory' in kwargs and kwargs['memory'] == True:
                loss_memory_Q, loss_memory_V = self.memory_loss(input_embeds.to(language_model_inputs.device),
                                                                language_model_inputs, ques_labels, cate_labels)
            else:
                loss_memory_Q, loss_memory_V = 0, 0

            # import pdb;pdb.set_trace()
            # update prototype
            self.update_prototype(current_prototype_Q, current_prototype_V, current_num_Q, current_num_V,
                                  current_task_id, proto_alpha, proto_beta)

            # retrieval the most relevant prototype
            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(
                input_embeds, dim=1), ques_labels)  # [bs, 768]
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)  # [bs, 1, 768]
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(
                language_model_inputs, dim=1), cate_labels)  # [bs, 768]
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)  # [bs, 1, 768]
        else:
            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(input_embeds, dim=1))  # [bs, 768]
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)  # [bs, 1, 768]
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(language_model_inputs, dim=1))  # [bs, 768]
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)  # [bs, 1, 768]
            loss_memory_Q, loss_memory_V = 0, 0


        # ========================================================
        inputs_embeds = torch.cat([language_model_inputs, input_embeds.to(language_model_inputs.device), 
            retrievaled_Q_proto.detach(), retrievaled_V_proto.detach()], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            B, L = attention_mask.size()
            VL = inputs_embeds.size(1) - L
            language_model_attention_mask = attention_mask.new_ones(B, VL).to(language_model_inputs.device)
        expected_device = language_model_attention_mask.device
        
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
       
        # inputs_embeds = self.get_input_embeddings()(input_ids)
        # retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(inputs_embeds, dim=1))  # [bs, 768]
        # retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)  # [bs, 1, 768]
        # retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(language_model_inputs, dim=1))  # [bs, 768]
        # retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)  # [bs, 1, 768]

        # inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device), 
        #     retrievaled_Q_proto.detach(), retrievaled_V_proto.detach()], dim=1)
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        #     B, L = attention_mask.size()
        #     VL = inputs_embeds.size(1) - L
        #     language_attention_mask = attention_mask.new_ones(B, VL).to(language_model_inputs.device)
        # expected_device = language_attention_mask.device
        
        # attention_mask = torch.cat([language_attention_mask, attention_mask.to(expected_device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs