import torch
from torch import nn
from models.base_model import load_base_model

import torch
from torch import nn
from models.base_model import load_base_model

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.in_dim = linear_layer.weight.size(1) 
        self.out_dim = linear_layer.weight.size(0)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        # froze weight and bias
        self.weight = nn.Parameter(linear_layer.weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(linear_layer.bias.clone(), requires_grad=False) if linear_layer.bias is not None else None

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(self.in_dim, rank)) # (in_dim, rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_dim)) # (rank, out_dim)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        frozen_out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B # [batch, in_dim] @ [in_dim, rank] @ [rank, out_dim]
        return frozen_out + (self.alpha / self.rank) * lora_out

def add_lora_to_model(model, rank=8, alpha=16.0):
    """Replace q_proj and v_proj nn.Linear layers with LoRALinear."""
    def get_parent_module(model, module_name):
        parts = module_name.split(".")
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        return current

    def get_child_name(module_name):
        return module_name.split(".")[-1]

    for name, module in list(model.named_modules()):
        if name.split(".")[-1] in ["q_proj", "v_proj"] and isinstance(module, nn.Linear):
            lora_module = LoRALinear(module, rank=rank, alpha=alpha)
            parent = get_parent_module(model, name)
            child_name = get_child_name(name)
            setattr(parent, child_name, lora_module)
            

    return model


def load_lora_applied_model(model_name, lora_checkpoint_path, rank=8, alpha=16.0):
    """
    Load a base model, inject LoRALinear layers into q_proj and v_proj,
    load LoRA weights from a checkpoint, and return tokenizer and model.
    """
    # Load base model and tokenizer
    tokenizer, base_model = load_base_model(model_name)

    # Inject LoRA layers
    lora_model = add_lora_to_model(base_model, rank=rank, alpha=alpha)

    # Load LoRA checkpoint
    lora_state_dict = torch.load(lora_checkpoint_path, map_location="cpu")

    # Update LoRA parameters
    with torch.no_grad():
        for key, param in lora_state_dict.items():
            try:
                module_path = key.split(".")[:-1]  # All but last part (e.g., "weight")
                param_name = key.split(".")[-1]
                module = lora_model
                for attr in module_path:
                    module = getattr(module, attr)
                getattr(module, param_name).copy_(param)
            except AttributeError:
                print(f"[WARNING] Skipping {key}: not found in model")

    # Log injected modules
    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"[INFO] LoRA injected at {name}: A={module.lora_A.shape}, B={module.lora_B.shape}")

    return tokenizer, lora_model