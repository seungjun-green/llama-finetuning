import os
from models.lora import LoRALinear
import torch

def save_lora_weights(model, save_directory, check_name):
    os.makedirs(save_directory, exist_ok=True)
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()

    torch.save(lora_state_dict, os.path.join(save_directory, check_name))
    

def save_checkpoint(model, dir, epoch, step, loss, global_min, block, total):
    ''' every log step save current lora weights
    '''
    print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
    save_lora_weights(model, dir, f"epoch{epoch}_step{step}_loss{loss.item():.4f}")