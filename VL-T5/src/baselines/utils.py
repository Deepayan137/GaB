def requires_grad(named_parameters):
    requires_grad_dict = {}
    for param in named_parameters:
        if param[1].requires_grad:
            requires_grad_dict.update({param[0]: param[1]})
    return requires_grad_dict
