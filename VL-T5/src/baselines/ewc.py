"""
Adapted from https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/ewc.py
"""


from collections import defaultdict
from typing import Dict, Tuple, Union
import warnings
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.utils import requires_grad

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData


        

class EWC(nn.Module):
    """
    Implements a simple Elastic Weight Consolidation strategy for updating the 
    weights of a model in a continual adaptation strategy. Implement the online setting
    where only the weights of previous step are preserved.

    """

    def __init__(
        self,
        ewc_lambda,
        decay_factor=None,
        uniform_importance=False
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param decay_factor: It specifies the decay term of the importance matrix.
        :param uniform_importance: Specifies whether to consider a uniform importance
                for all the weights of the network
        """

        super().__init__()

        self.ewc_lambda = ewc_lambda
        self.decay_factor = decay_factor
        self.uniform_importance = uniform_importance
        
        self.saved_params: Dict[str, ParamData] = defaultdict(dict)
        self.importances: Dict[str, ParamData] = defaultdict(dict)
        

    def before_backward(self, model, **kwargs):
        """
        Compute EWC penalty to add it the loss.
        """
        device = kwargs['device']
        penalty = torch.tensor(0).float().to(device)

        for k, cur_param in model.named_parameters():
            # new parameters do not count or first step
            if k not in self.saved_params:
                continue
            
            # weights that do not require grad are not considered
            if k not in self.importances:
                continue
            saved_param = self.saved_params[k]
            imp = self.importances[k]
            new_shape = cur_param.shape
            penalty += (
                imp.expand(new_shape)
                * (cur_param - saved_param.expand(new_shape)).pow(2)
            ).sum()

        return self.ewc_lambda * penalty

    def after_training_exp(self, model, optimizer, dloader, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if self.uniform_importance and len(self.importances) == 0:
            importances = zerolike_params_dict(model)
            requires_grad_keys = requires_grad(model.named_parameters()).keys()
            importances = dict(filter(lambda it: it[0] in requires_grad_keys, importances.items()))
            for k, param in importances.items():
                param = param.data + 1
            self.importances = importances
        else: 
            importances = self.compute_importances(
                model,
                optimizer,
                dloader, **kwargs
            )
            self.update_importances(importances)
        self.saved_params = copy_params_dict(model)

    def compute_importances(
        self, model, optimizer, dataloader, **kwargs) -> Dict[str, ParamData]:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # list of list
        importances = zerolike_params_dict(model)
        requires_grad_keys = requires_grad(model.named_parameters()).keys()
        importances = dict(filter(lambda it: it[0] in requires_grad_keys, importances.items()))
        for i, batch in enumerate(tqdm(dataloader, desc='Computing weights relevance')):

            optimizer.zero_grad()
            out = model.train_step(batch=batch, **kwargs)
            loss = out['loss']
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                requires_grad(model.named_parameters()).items(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances

    @torch.no_grad()
    def update_importances(self, importances):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

 
        for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
            self.importances.items(),
            importances.items(),
            fillvalue=(None, None),
        ):
            # Add new module importances to the importances value (New head)
            if k1 is None:
                assert k2 is not None
                assert curr_imp is not None
                self.importances[k2] = curr_imp
                continue

            assert k1 == k2, "Error in importance computation."
            assert curr_imp is not None
            assert old_imp is not None
            assert k2 is not None

            # manage expansion of existing layers
            self.importances[k1] = ParamData(
                f"imp_{k1}",
                curr_imp.shape,
                init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                + curr_imp.data,
                device=curr_imp.device,
            )


ParamDict = Dict[str, Union[ParamData]]
EwcDataType = Tuple[ParamDict, ParamDict]

from torchvision.models import resnet18
from tqdm import tqdm

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
    model[2].weight.requires_grad_(False)

    model = model.to(device)
    
    ewc_module = EWC(ewc_lambda=1, decay_factor=0.99, uniform_importance=True)
    loss = ewc_module.before_backward(model, device=torch.device('cpu'))
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
            loss += ewc_module.before_backward(model, device=device)
            loss.backward()
            optimizer.step()

    ewc_module.after_training_exp(model, criterion, optimizer, dloader, device)
        
    iter_dloader = iter(dloader)
    for epoch_idx in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(iter_dloader):
            optimizer.zero_grad()
            in_data, label = batch
            in_data = in_data.to(device)
            label = label.to(device)

            pred = model(in_data)
            loss = criterion(pred, label)
            ewc_loss = ewc_module.before_backward(model, device=device)
            loss = loss + ewc_loss
            loss.backward()
            optimizer.step()
    ewc_module.after_training_exp(model, criterion, optimizer, dloader, device)