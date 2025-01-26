import os
from models.lora import LoRALinear
import torch

def save_lora_weights(model, save_directory, check_name):
    os.makedirs(save_directory, exist_ok=True)
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_a.weight"] = module.lora_a.weight.detach().cpu()
            lora_state_dict[f"{name}.lora_b.weight"] = module.lora_b.weight.detach().cpu()

    torch.save(lora_state_dict, os.path.join(save_directory, check_name))
    

def save_checkpoint(model, dir, epoch, step, loss, global_min, block, total):
    if  step % block == 0 and step != 0 or step == total - 1:
        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
        save_lora_weights(model, dir, f"epoch{epoch}_step{step}_loss{loss.item():.4f}")

    # save the global min 
    if loss.item() < global_min:
        save_lora_weights(model, dir, f"global_min_epoch{epoch}_step{step}_loss{loss.item():.4f}")
        
        global_min = loss.item()
    
    return global_min