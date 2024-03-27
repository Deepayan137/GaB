"""
    Adapted from https://github.com/JH-LEE-KR/l2p-pytorch/blob/main/prompt.py
    and https://github.com/JH-LEE-KR/l2p-pytorch/blob/main/vision_transformer.py
"""
import torch
import torch.nn as nn
# wrapper to feature extractor
# add prompts codebook
# forward features
# get topk prompts
# prepend topk prompts
# feedforward to pretrained model
# predict output

import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompt'] = batched_prompt

        return out

class Learning2Prompt(nn.Module):
    def __init__(self, pull_constraint=True, length=5, embed_dim=768, pool_size=None, **kwargs):
        super().__init__()
        self.prompt = Prompt(length=length, 
                             embed_dim=embed_dim, 
                             pool_size=pool_size,
                             **kwargs)
        self.task_id = 0
        self.pull_constraint = pull_constraint
        self.pull_constraint_coeff = 0.5
        self.use_prompt_mask =True
    
    def before_backward(self, model, x_input, cls_features=None, **kwargs):
        """
        Compute prompt learning loss
        """
        feat = model.get_patch_embeddings(x_input)
        
        if self.use_prompt_mask and self.train:
            start = self.task_id * self.prompt.top_k
            end = (self.task_id + 1) * self.prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(x_input.device)
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(x_input.shape[0], -1)
            if end > self.prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        res = self.prompt(feat, prompt_mask=prompt_mask, cls_features=cls_features)
        self.total_prompt_len = res['total_prompt_len']

        
        if self.pull_constraint and 'reduce_sim' in res:
            loss = - self.pull_constraint_coeff * res['reduce_sim']
        res.update({'loss': loss})
        return res
    
    def after_training_exp(self, **kwargs):
        """
        Compute update the set of prompt to be used
        """
        self.task_id += 1




from torchvision.models import vit_b_16
from tqdm import tqdm
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import timm
from einops import rearrange

def forward_head(classifier, x):
    x = x[:, 1:(1 + 5)]
    x = x.mean(dim=1)
    return x

class ToyModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        model = timm.create_model(
        f"hf-hub:timm/vit_giant_patch14_224", pretrained=False,  global_pool=None
        )
        self.patch_embed = model.patch_embed.proj
        self.blocks = model.blocks
        self.prompt_len = 5
        self.fc = nn.Linear(self.prompt_len * 5 * 768, 10)

    def forward(self, x):
        x = rearrange(x, 'b c (kh h) (kw w) -> (b kh kw) c h w', kh=14, kw=14)
        x = self.patch_embed(x)
        x = rearrange(x, '(b kh kw) c h w -> b (kh kw) (c h w)', kh=14, kw=14)
        return x
    
    def classify(self, x):
        x = self.blocks(x)
        x = self.fc(x[:, :self.prompt_len* 5].flatten(1, 2))

        return x
    def get_patch_embeddings(self, x):
        x = rearrange(x, 'b c (kh h) (kw w) -> (b kh kw) c h w', kh=14, kw=14)
        x = self.patch_embed(x)
        x = rearrange(x, '(b kh kw) c h w -> b (kh kw) (c h w)', kh=14, kw=14)
        return x

        
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 10
    num_epochs = 5 
    
    model = ToyModel()
    model = model.to(device)
    ewc_module = Learning2Prompt(prompt_key=True, pool_size=15, top_k=5, prompt_pool=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # dataset and dloader
    dataset = FakeData(size = 10000, 
                                  image_size = (3, 224, 224), 
                                  num_classes = 10, 
                                  transform = ToTensor())   
    dloader = torch.utils.data.DataLoader(dataset, batch_size = 128)
    iter_dloader = iter(dloader)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(tqdm(iter_dloader, leave=False)):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            res = ewc_module.before_backward(model, in_data, device=device)
            loss = res['loss']
            prompt = res['prompt']
            prompted_feat = torch.cat([prompt, model.get_patch_embeddings(in_data)], dim=1)
            pred = model.classify(prompted_feat)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()

    ewc_module.after_training_exp()
        
    iter_dloader = iter(dloader)
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(tqdm(iter_dloader, leave=False)):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            res = ewc_module.before_backward(model, in_data, device=device)
            loss = res['loss']
            prompt = res['prompt']
            prompted_feat = torch.cat([prompt, model.get_patch_embeddings(in_data)], dim=1)
            pred = model.classify(prompted_feat)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
    
    ewc_module.after_training_exp()
        
    iter_dloader = iter(dloader)
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(tqdm(iter_dloader, leave=False)):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            res = ewc_module.before_backward(model, in_data, device=device)
            loss = res['loss']
            prompt = res['prompt']
            prompted_feat = torch.cat([prompt, model.get_patch_embeddings(in_data)], dim=1)
            pred = model.classify(prompted_feat)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
