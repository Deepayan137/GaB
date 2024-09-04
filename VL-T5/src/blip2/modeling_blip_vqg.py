import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.models.blip_2.modeling_blip_2 import (Blip2VisionModel,
    Blip2ForConditionalGeneration, Blip2QFormerLayer, Blip2QFormerEncoder, Blip2QFormerModel, Blip2ForConditionalGenerationModelOutput)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions, 
    BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPooling)


class NaiveBlip2VQACL(Blip2ForConditionalGeneration):
    def __init__(self, config, pool_size=None, prompt_pool=False):
        super().__init__(config)
        self.vision_model = Blip2VisionModelOurs(
            config.vision_config, 
            pool_size=pool_size, 
            prompt_pool=prompt_pool)
        self.qformer = Blip2QFormerModelOurs(config.qformer_config)
        self.language_projection_answers = nn.Linear(config.qformer_config.hidden_size, 
            config.text_config.hidden_size)

        self.language_projection_questions = nn.Linear(config.qformer_config.hidden_size, 
            config.text_config.hidden_size)
        self.answer_attention = MLP(2*config.text_config.hidden_size, att_ff_size, hidden_size,
                                    num_layers=num_att_layers)
        self.mu_answer_encoder = nn.Linear(hidden_size, z_size)
        self.logvar_answer_encoder = nn.Linear(hidden_size, z_size)
        self.z_decoder = nn.Linear(z_size, hidden_size)
        
    # @torch.no_grad()  # Ensure that gradients are not calculated for this operation
    def get_features(self, pixel_values):
        """
        Extract features from the vision model and the query transformer.

        Args:
            pixel_values (torch.Tensor): The pixel values of the images to process.

        Returns:
            torch.Tensor: The features extracted by the query transformer.
        """
        # Ensure the model is in eval mode, which is standard practice when not training
        # self.eval()

        # Forward pass through the vision model to get image embeddings
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state

        # Forward pass through the query transformer
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )


        # Switch back to training mode
        # self.train()

        return query_outputs, vision_outputs

    def reconstruct_inputs(self, image_features, answer_features):
        """Reconstructs the image features using the VAE.

        Args:
            image_features: Batch of image features.
            answer_features: Batch of answer features.

        Returns:
            Reconstructed image features and answer features.
        """
        recon_image_features = None
        recon_answer_features = None
        mus, logvars = self.encode_into_z(image_features, answer_features)
        zs = self.reparameterize(mus, logvars)
        if self.image_recon:
            recon_image_features = self.image_reconstructor(zs)
        if self.answer_recon:
            recon_answer_features = self.answer_reconstructor(zs)
        return recon_image_features, recon_answer_features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode_into_z(self, inputs_embeds):
        """Encodes the attended features into z space.

        Args:
            image_features: Batch of image features.
            answer_features: Batch of answer features.

        Returns:
            mus and logvars of the batch.
        """
        attended_hiddens = self.answer_attention(inputs_embeds)
        mus = self.mu_answer_encoder(attended_hiddens)
        logvars = self.logvar_answer_encoder(attended_hiddens)
        return mus, logvars

    def forward(self,
        query_outputs,
        vision_outputs,
        input_ids,
        mode,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        query_output = query_outputs[0]
        # step 3: use the language model, conditioned on the query outputs and the prompt
        if mode == 'answers':
            language_model_inputs = self.language_projection_answers(query_output)
        elif mode == 'questions':
            language_model_inputs = self.language_projection_questions(query_output)
        else:
            raise ValueError("Mode must be answers or questions ")
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
        query_outputs,
        vision_outputs,
        input_ids=None,
        attention_mask=None,
        mode='answers',
        **generate_kwargs):

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        query_output = query_outputs[0]
        batch_size = query_output.shape[0]
        query_output = query_outputs.last_hidden_state
        image_embeds = vision_outputs.last_hidden_state
        if mode == 'answers':
            language_model_inputs = self.language_projection_answers(query_output)
        elif mode == 'questions':
            language_model_inputs = self.language_projection_questions(query_output)
        else:
            raise ValueError("Mode must be answers or questions")
        # language_model_inputs = self.language_projection(query_output)
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

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        if not self.language_model.config.is_encoder_decoder:
            bos_tokens = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)
        return outputs