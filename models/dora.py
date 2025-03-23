import torch
from torch import nn
from models.base_model import load_base_model


class DoRALinear(nn.Module):
    def __init__(self, original_linear, r):
        super().__init__()
        self.original_linear = original_linear
        self.d, self.k = original_linear.weight.shape

        m = torch.norm(original_linear.weight, dim=0, keepdim=True)
        self.dora_m = nn.Parameter(m)  # [1, k]
        self.V = original_linear.weight.clone().detach() # [d, k]
        # initalize vlaues of B as zeros and values of A to
        # follow kaiming distribution as shown in the paper
        self.dora_B = nn.Parameter(torch.zeros(self.d, r))  # [d, r]
        self.dora_A = nn.Parameter(torch.empty(r, self.k))  # [r, k]
        nn.init.kaiming_uniform_(self.dora_A)
        
        # set the bias
        self.bias = original_linear.bias if original_linear.bias is not None else None

    def forward(self, x):
        delta_V = self.dora_B @ self.dora_A  # [d, r] @ [r, k] -> [d, k]
        V_prime = self.V.to(delta_V.device) + delta_V  # [d, k]
        norm_V_prime = torch.norm(V_prime, dim=0, keepdim=True).detach()  # [1, k]
        W_prime = self.dora_m * (V_prime / norm_V_prime)  # [d, k]
        # set a linear layer with the W_prime and do the forward propagation with x
        return torch.nn.functional.linear(x, W_prime, self.bias)


def add_dora_to_model(model, targets=['q_proj', 'v_proj'], rank=8):
    def get_parent_module(model, module_name):
        parts = module_name.split(".")
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        return current

    def get_child_name(module_name):
        return module_name.split(".")[-1]
    
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] in targets and isinstance(module, nn.Linear):
            dora_module = DoRALinear(module, r=rank)
            parent = get_parent_module(model, name)
            child = get_child_name(name)
            setattr(parent, child, dora_module)
            
    return model