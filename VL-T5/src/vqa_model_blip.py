from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.modeling_blip import Blip2VQACL
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGeneration

from typing import Tuple, Optional
from transformers.models.blip_2.modeling_blip_2 import (
	Blip2ForConditionalGeneration, Blip2QFormerModel, 
    Blip2ForConditionalGenerationModelOutput, Blip2PreTrainedModel, Blip2VisionConfig,
    Blip2VisionEmbeddings, Blip2EncoderLayer)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.utils import logging
from typing import Union
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn import CrossEntropyLoss
from transformers.models.blip_2.configuration_blip_2 import Blip2Config

logger = logging.get_logger(__name__)

class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None, 
        embeddings=None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        Prepare processor, model and image input

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        ... )  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ```

        Image captioning (without providing a text prompt):

        ```python
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```

        Visual question answering (prompt = question):

        ```python
        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```

        Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
        This greatly reduces the amount of memory used by the model while maintaining the same performance.

        ```python
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
        ... )  # doctest: +IGNORE_RESULT

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embeddings=embeddings
        )

        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # if embeddings is not None:
            # image_embeds = image_embeds[:pixel_values.shape[0]]
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

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
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
                loss_fct = CrossEntropyLoss(reduction="mean")

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
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
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
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1]
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/blip_2/modeling_blip_2.py#L471
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """

    def __init__(self, config: Blip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/blip_2/modeling_blip_2.py#L501
class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()
        self.learnable_prompt = nn.Parameter(torch.randn(80, 252, 1408))

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embeddings: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states = self.embeddings(pixel_values)
        if embeddings is not None:
            hidden_states = torch.cat([embeddings, hidden_states], dim=1)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # breakpoint()
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if embeddings is not None:
            pooled_output = last_hidden_state[:, :embeddings.shape[1]+1].mean(dim=1) 
        else:
            pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings

class NaiveBLIP2(Blip2ForConditionalGeneration):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        from transformers import AutoProcessor
        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        print("Freezing vision model")
        for name, param in self.vision_model.named_parameters():
            param.requires_grad = False
        
        self.vision_model = self.vision_model.eval()
        # print("Freezing qformer")
        # for name, param in self.qformer.named_parameters():
        #     param.requires_grad = False
        
        print("Freezing qformer")
        num_layers = len(self.qformer.encoder.layer)

        # for i, layer in enumerate(self.qformer.encoder.layer):
        #     # Freeze all layers except the last one
        #     if i < num_layers - 1:
        for param in self.qformer.parameters():
            param.requires_grad = False
        self.qformer.eval()

        # self.query_tokens.requires_grad = True
        # print("freeze vision encoder")
        
        for name, param in self.language_model.named_parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            param.requires_grad = False
        print("Freezing language model but lm_head") 
        for name, param in self.language_projection.named_parameters():
            param.requires_grad = True
        for name, param in self.language_model.lm_head.named_parameters():
            param.requires_grad = True
 
        # self.eos_token_id = self.processor.tokenizer('\n', add_special_tokens=False).input_ids[0]
    
    def get_patch_embeddings(self, x):
        device = next(self.parameters()).device
        pixel_values = x.to(device)
        emb = self.vision_model.embeddings(pixel_values)
        return emb

    @torch.no_grad()
    def test_step(self, batch, task, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]
        batch_size = pixel_values.shape[0]
        
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)

        # cate_labels = batch['cate_labels'].to(device)
        # ques_labels = batch['ques_labels'].to(device)

        # qa pairs generation
        # qa_pairs = torch.cat([input_ids, lm_labels], dim=1) ## gt (q,a) pairs as labels
        # prompt = f'Generate question answer pairs on task {task}'
        # input_ids = self.processor(None, text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=12)['input_ids'].repeat(batch_size, 1).to(device)
        # max_new_tokens = 25
        # max_new_tokens = 2
        # if task in ['q_recognition', 'q_type']:
            # max_new_tokens = 3
        # output = self.generate(pixel_values=pixel_values, max_length=25, do_sample=True, num_beams=25, temperature=2.0, num_return_sequences=5, top_p=5)
        output = self.generate(
                        pixel_values=pixel_values, 
                        max_length=25, 
                        do_sample=False, 
                        num_beams=27, 
                        num_return_sequences=3, 
                        diversity_penalty=0.5, 
                        num_beam_groups=9)
        # output = self.generate(input_ids=input_ids,
            # pixel_values=pixel_values,
            # max_new_tokens=10,
            # num_beams=5,
            # temperature=0.7,
            # repetition_penalty=2.5, 
            # do_sample=True)
        result = {}
        result['token_ids'] = output
        
        # decoded_prompt = self.processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print("prompt", decoded_prompt[:3])
        result['pred_ans'] = self.processor.batch_decode(output, 
                                    skip_special_tokens=True, 
                                    )
        # qa_pairs = self.processor.tokenizer.batch_decode(qa_pairs, skip_special_tokens=True)
        # print("qa_pair", qa_pairs[:3])
        print("pred_answer: ", result['pred_ans'][:5])
        return result

    def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False, embeddings=None, **kwargs):
        device = next(self.parameters()).device
        pixel_values = batch['pixel_values'].to(device) # bs, 36, 2048
        input_ids = batch['input_ids'].to(device) # bs, 20
        lm_labels = batch["target_ids"].to(device) #[bs, 5]
        batch_size = pixel_values.shape[0]
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
        
        # qa pairs generation
        # qa_pairs = torch.cat([input_ids, lm_labels], dim=1) ## gt (q,a) pairs as labels
        # task = kwargs['task']
        # prompt = [f"Task {task}:" for _ in range(batch_size)]
        # prompt_ids = self.processor(text=prompt, 
                                    # max_length=5, 
                                    # padding=True, 
                                    # truncation=True, 
                                    # return_tensors="pt")['input_ids'].to(device)

        # target_ids = self.processor(text=sent, 
                                    # max_length=25, 
                                    # truncation=True, 
                                    # padding=True, 
                                    # return_tensors="pt")['input_ids']

        output = self(input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            )
        

        assert 'loss' in output
        B, L = input_ids.size()
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

        self.vision_model = Blip2VisionModel(config.vision_config)
        super().__init__(config)
        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q = 0, total_num_Q = 1000, memory=False, embeddings=None):

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
            return_dict=True,
            embeddings=embeddings,
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

