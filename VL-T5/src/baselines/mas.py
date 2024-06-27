"""
Adapted from https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/mas.py
"""
from tqdm.auto import tqdm
from typing import Dict, Union

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from src.baselines.utils import requires_grad

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData

# TODO: uniform baselines naming - importances/importance compute_importances/_get_importances
class MAS(nn.Module):
    """
    Memory Aware Synapses (MAS) plugin.

    Similarly to EWC, the MAS plugin computes the importance of each
    parameter at the end of each experience. The approach computes
    importance via a second pass on the dataset. MAS does not require
    supervision and estimates importance using the gradients of the
    L2 norm of the output. Importance is then used to add a penalty
    term to the loss function.

    Technique introduced in:
    "Memory Aware Synapses: Learning what (not) to forget"
    by Aljundi et. al (2018).

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/mas.py
    """

    def __init__(self, lambda_reg: float = 1.0, alpha: float = 0.5, verbose=False):
        """
        :param lambda_reg: hyperparameter weighting the penalty term
               in the loss.
        :param alpha: hyperparameter used to update the importance
               by also considering the influence in the previous
               experience.
        :param verbose: when True, the computation of the influence
               shows a progress bar using tqdm.
        """

        # Init super class
        super().__init__()

        # Regularization Parameters
        self._lambda = lambda_reg
        self.alpha = alpha

        # Model parameters
        self.params: Union[Dict, None] = None
        self.importance: Union[Dict, None] = None

        # Progress bar
        self.verbose = verbose

    def _get_importance(self, model, optimizer, dloader, **kwargs):
        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model))
        requires_grad_keys = requires_grad(model.named_parameters()).keys()
        importance = dict(filter(lambda it: it[0] in requires_grad_keys, importance.items()))


        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        # Progress bar
        if self.verbose:
            print("Computing importance")
            dloader = tqdm(dloader)

        for _, batch in enumerate(dloader):

            # Forward pass
            optimizer.zero_grad()
            out = model.train_step(batch, **kwargs)
            out = out['logits']

            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # In multi-head architectures, the gradient is going
                    # to be None for all the heads different from the
                    # current one.
                    if param.grad is not None:
                        importance[name].data += param.grad.abs()

        # Normalize importance
        for k in importance.keys():
            importance[k].data /= float(len(dloader))

        return importance

    def before_backward(self, model, **kwargs):
        # Check if the task is not the first
        if self.importance is None:
            for idx, p in enumerate(model.parameters()):
                device = p.device
                break
            return torch.tensor(0.0, device = device)

        loss_reg = 0.0

        # Check if properties have been initialized
        if not self.importance:
            raise ValueError("Importance is not available")
        if not self.params:
            raise ValueError("Parameters are not available")
        # Apply penalty term
        for name, param in model.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(
                    self.importance[name].expand(param.shape)
                    * (param - self.params[name].expand(param.shape)).pow(2)
                )

        # Update loss
        return self._lambda * loss_reg

    def after_training_exp(self, model, optimizer, dloader, **kwargs):
        params = dict(copy_params_dict(model))
        requires_grad_keys = requires_grad(model.named_parameters()).keys()
        self.params = dict(filter(lambda it: it[0] in requires_grad_keys, params.items()))

        # Get importance
        if self.importance is None:
            self.importance = self._get_importance(model, optimizer, dloader, **kwargs)
            return
        else:
            curr_importance = self._get_importance(model, optimizer, dloader, **kwargs)

        # Check if previous importance is available
        if not self.importance:
            raise ValueError("Importance is not available")

        # Update importance
        for name in curr_importance.keys():
            new_shape = curr_importance[name].data.shape
            if name not in self.importance:
                self.importance[name] = ParamData(
                    name,
                    curr_importance[name].shape,
                    device=curr_importance[name].device,
                    init_tensor=curr_importance[name].data.clone(),
                )
            else:
                self.importance[name].data = (
                    self.alpha * self.importance[name].expand(new_shape)
                    + (1 - self.alpha) * curr_importance[name].data
                )
    def get_state_dict(self):
        if self.params is None or self.importance is None:
            return {}
        params_dict = {}
        for p in self.params.keys():
            params_dict[p] = self.params[p].data

        importance_dict = {}
        for p in self.importance.keys():
            importance_dict[p] = self.importance[p].data
        
        s_dict = {'params': params_dict,
                    'importance': importance_dict}
        return s_dict
    
    def load_state_dict(self, state_dict):
        # load params
        self.params = dict()
        
        for name in state_dict['params'].keys():
            new_shape = state_dict['params'][name].data.shape
            if name not in self.params:
                self.params[name] = ParamData(
                    name,
                    state_dict['params'][name].shape,
                    device=state_dict['params'][name].device,
                    init_tensor=state_dict['params'][name].data.clone(),
                )

        # load importance
        self.importance = dict()
        for name in state_dict['importance'].keys():
            new_shape = state_dict['importance'][name].data.shape
            if name not in self.importance:
                self.importance[name] = ParamData(
                    name,
                    state_dict['importance'][name].shape,
                    device=state_dict['importance'][name].device,
                    init_tensor=state_dict['importance'][name].data.clone(),
                )


from torchvision.models import resnet18
from tqdm import tqdm
import torch.nn as nn
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 10
    num_epochs = 5 
    model = nn.Sequential(nn.Linear(1, 32),
                          nn.ReLU(),
                          nn.Linear(32, 32),
                          nn.ReLU(),
                          nn.Linear(32, num_classes)
                        )
    
    model.requires_grad_(False)
    model[0].weight.requires_grad_(True)
    model = model.to(device)
    
    mas_module = MAS(lambda_reg=1, alpha=0.99)
    loss = mas_module.before_backward(model, device=torch.device('cpu'))
    dataset_means = torch.arange(num_classes).float()
    dataset_stds = torch.rand(len(dataset_means))* 0.3
    gaussians = [torch.distributions.Normal(dataset_means[idx], dataset_stds[idx]) for idx in range(len(dataset_means))]

    num_points = 1000
    data = [gaussians[idx].sample((num_points,)) for idx in range(len(gaussians))]
    labels = [torch.full((1000, ), l) for l in range(len(gaussians))]
    labels = torch.cat(labels, dim=0)
    data = torch.cat(data, dim=0)
    
    criterion = nn.CrossEntropyLoss()
    

    dataset = torch.utils.data.TensorDataset(data[:num_points][:, None], labels[:num_points])
    dloader = torch.utils.data.DataLoader(dataset, batch_size = 16)
    iter_dloader = iter(dloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(iter_dloader):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            pred = model(in_data)
            loss = criterion(pred, label)
            loss += mas_module.before_backward(model)
            loss.backward()
            optimizer.step()

    mas_module.after_training_exp(model, dloader, optimizer,  device)
        
    iter_dloader = iter(dloader)
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(iter_dloader):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            pred = model(in_data)
            loss = criterion(pred, label)
            ewc_loss = mas_module.before_backward(model, device=device)
            loss = loss + ewc_loss
            loss.backward()
            optimizer.step()