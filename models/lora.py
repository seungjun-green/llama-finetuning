import torch
from torch import nn
from models.base_model import load_base_model

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=16.0, rank=8, dropout=0.0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        frozen_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return frozen_out + (self.alpha / self.rank) * lora_out

def add_lora_to_model(model, rank=8, alpha=16.0):
    # replace the selected nn.Linear layers (e.g q_proj, v_proj) with LoRA layers.
    def get_parent_module(model, module_name):
        parts = module_name.split(".")
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        return current

    def get_child_name(module_name):
        return module_name.split(".")[-1]

    for name, module in list(model.named_modules()):
        if "q_proj" in name or "v_proj" in name:
            if not isinstance(module, nn.Linear):
                continue

            in_dim = module.weight.size(1)
            out_dim = module.weight.size(0)
            bias = module.bias is not None

            lora_module = LoRALinear(in_dim, out_dim, alpha=alpha, rank=rank, bias=bias)
            with torch.no_grad():
                lora_module.linear.weight.copy_(module.weight)
                if bias:
                    lora_module.linear.bias.copy_(module.bias)

            parent = get_parent_module(model, name)
            child_name = get_child_name(name)
            setattr(parent, child_name, lora_module)

    return model




def load_lora_applied_model(model_name: str, lora_checkpoint_path: str, rank=8, alpha=16.0):
    """
    1. Load base model & tokenizer.
    2. Inject LoRALinear layers into q_proj & v_proj.
    3. Load LoRA weights into those layers.
    4. Return (tokenizer, lora_model).
    """
    # 1. Load the base
    tokenizer, base_model = load_base_model(model_name)

    # 2. Inject LoRA layers
    lora_model = add_lora_to_model(base_model, rank=rank, alpha=alpha)

    # 3. Load LoRA checkpoint
    lora_state_dict = torch.load(lora_checkpoint_path, map_location="cpu")

    # 4. Copy parameters
    with torch.no_grad():
        for key, param in lora_state_dict.items():
            splitted = key.split(".")
            # everything except the last two is the module path
            module_path = splitted[:-2]
            # second-last is "lora_a" or "lora_b"
            submodule_name = splitted[-2]
            # last is "weight" (most likely)
            weight_name = splitted[-1]

            module = lora_model
            for attr in module_path:
                module = getattr(module, attr)

            submodule = getattr(module, submodule_name)
            getattr(submodule, weight_name).copy_(param)

    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"[INFO] LoRA module injected at: {name}")

    return tokenizer, lora_model