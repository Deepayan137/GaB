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
from src.blip2.modeling_blip import Blip2VisionModelOurs, Blip2QFormerModelOurs

"""A simple MLP.
"""

from collections import OrderedDict
from torch import nn

import math


class MLP(nn.Module):
	"""A simple MLP.
	"""

	def __init__(self, input_size, hidden_size, num_classes,
				 num_layers=1, dropout_p=0.0):
		"""Constructor for MLP.

		Args:
			input_size: The number of input dimensions.
			hidden_size: The number of hidden dimensions for each layer.
			num_classes: The size of the output.
			num_layers: The number of hidden layers.
			dropout_p: Dropout probability.
		"""
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		layers = []
		for i in range(num_layers):
			idim = hidden_size
			odim = hidden_size
			if i == 0:
				idim = input_size
			if i == num_layers-1:
				odim = num_classes
			fc = nn.Linear(idim, odim)
			fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
			fc.bias.data.fill_(0)
			layers.append(('fc'+str(i), fc))
			if i != num_layers-1:
				layers.append(('relu'+str(i), nn.ReLU()))
				layers.append(('dropout'+str(i), nn.Dropout(p=dropout_p)))
		self.layers = nn.Sequential(OrderedDict(layers))

	def params_to_train(self):
		return self.layers.parameters()

	def forward(self, x):
		"""Propagate through all the hidden layers.

		Args:
			x: Input of self.input_size dimensions.
		"""
		out = self.layers(x)
		return out

class NaiveBlip2VQG(Blip2ForConditionalGeneration):
	def __init__(self, config, pool_size=None, prompt_pool=False):
		super().__init__(config)
		self.vision_model = Blip2VisionModelOurs(
			config.vision_config, 
			pool_size=pool_size, 
			prompt_pool=prompt_pool)
		self.qformer = Blip2QFormerModelOurs(config.qformer_config)
		self.language_projection = nn.Linear(config.qformer_config.hidden_size, 
			config.text_config.hidden_size)
		z_size=512
		self.answer_attention = MLP(2*config.text_config.hidden_size, z_size, config.text_config.hidden_size,
									num_layers=2)
		self.mu_answer_encoder = nn.Linear(config.text_config.hidden_size, z_size)
		self.logvar_answer_encoder = nn.Linear(config.text_config.hidden_size, z_size)
		self.z_decoder = nn.Linear(z_size, config.text_config.hidden_size)
		self.answer_reconstructor = MLP(z_size, 512, config.text_config.hidden_size,num_layers=2)
		self.pool = nn.AdaptiveAvgPool1d(42)
	# @torch.no_grad()  # Ensure that gradients are not calculated for this operation
	def encode_images(self, pixel_values):
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

		query_output = query_outputs[0]
		# Switch back to training mode
		# self.train()
	
		language_model_inputs = self.language_projection(query_output)
		language_model_attention_mask = torch.ones(
			language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
		)
		return language_model_inputs, language_model_attention_mask, vision_outputs, query_outputs

	def encode_answers(self, input_ids):
		return self.language_model.get_input_embeddings()(input_ids)

	def encode_into_z(self, image_features, answer_features):
		pool = nn.AdaptiveAvgPool1d(answer_features.size(1))
		image_features = pool(image_features.permute(0, 2, 1)).permute(0, 2, 1)
		answer_features = pool(answer_features.permute(0, 2, 1)).permute(0, 2, 1)

		together = torch.cat([image_features, answer_features], dim=2)
		attended_hiddens = self.answer_attention(together)
		mus = self.mu_answer_encoder(attended_hiddens)
		logvars = self.logvar_answer_encoder(attended_hiddens)
		return mus, logvars

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

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
		num_patches = image_features.size(1)
		mus, logvars = self.encode_into_z(image_features, answer_features)
		zs = self.reparameterize(mus, logvars)
		recon_answer_features = self.answer_reconstructor(zs)
		return recon_image_features, recon_answer_features

	def gaussian_KL_loss(self, mus, logvars, eps=1e-8):
		"""Calculates KL distance of mus and logvars from unit normal.

		Args:
			mus: Tensor of means predicted by the encoder.
			logvars: Tensor of log vars predicted by the encoder.

		Returns:
			KL loss between mus and logvars and the normal unit gaussian.
		"""
		KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
		kl_loss = KLD/(mus.size(0) + eps)
		"""
		if kl_loss > 100:
			print kl_loss
			print KLD
			print mus.min(), mus.max()
			print logvars.min(), logvars.max()
			1/0
		"""
		return kl_loss

	def info_parameters(self):
		params = (list(self.answer_attention.parameters()) +
				  list(self.mu_answer_encoder.parameters()) +
				  list(self.logvar_answer_encoder.parameters()))

		# Reconstruction parameters.
		# if self.image_recon:
		# 	params += list(self.image_reconstructor.parameters())
		
		params += list(self.answer_reconstructor.parameters())

		params = filter(lambda p: p.requires_grad, params)
		return params

	def generator_parameters(self):
		params = self.parameters()
		params = filter(lambda p: p.requires_grad, params)
		return params

	def decode_questions(self,
		inputs_embeds,
		vision_outputs,
		query_outputs,
		attention_mask,
		labels
		):
		return_dict =True
		outputs = self.language_model(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
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
		
		return Blip2ForConditionalGenerationModelOutput(
			loss=loss,
			logits=logits,
			vision_outputs=vision_outputs,
			qformer_outputs=query_outputs,
			language_model_outputs=outputs,
		)

	def forward(self,
		pixel_values,
		input_ids,
		labels,
		attention_mask=None,
		decoder_input_ids=None,
		decoder_attention_mask=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None):

		return_dict = True
		# step 3: use the language model, conditioned on the query outputs and the prompt
		language_model_inputs, language_model_attention_mask, vision_outputs, query_outputs = self.encode_images(pixel_values)
		answer_features = self.encode_answers(labels)
		#####
		mus, logvars = self.encode_into_z(language_model_inputs, answer_features.to(language_model_inputs.device))
		zs = self.reparameterize(mus, logvars)
		inputs_embeds =  self.z_decoder(zs)
		#####
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		expected_device = language_model_attention_mask.device
		attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
		return self.decode_questions(answer_features, attention_mask, input_ids)

	@torch.no_grad()
	def generate(
		self,
		pixel_values=None,
		input_ids=None,
		attention_mask=None,
		**generate_kwargs):

		if hasattr(self, "hf_device_map"):
			# preprocess for `accelerate`
			self._preprocess_accelerate()
		language_model_inputs, language_model_attention_mask, vision_outputs, query_outputs = self.encode_images(pixel_values)
		image_embeds = vision_outputs.last_hidden_state
		
		inputs_embeds = self.encode_answers(input_ids)
		inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
		batch_size = input_ids.shape[0]
		
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1)
		#####
		
		mus, logvars = self.encode_into_z(language_model_inputs, inputs_embeds.to(language_model_inputs.device))
		zs = self.reparameterize(mus, logvars)
		inputs_embeds =  self.z_decoder(zs)
		#####
		outputs = self.language_model.generate(
			inputs_embeds=inputs_embeds,
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