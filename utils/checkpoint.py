import os
from models.lora import LoRALinear
import torch

def save_lora_weights(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_a.weight"] = module.lora_a.weight.detach().cpu()
            lora_state_dict[f"{name}.lora_b.weight"] = module.lora_b.weight.detach().cpu()

    torch.save(lora_state_dict, os.path.join(save_directory, "lora_weights.pt"))
    
    
def checkponit(model, dir, epoch, step, loss, global_min):
    print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
    # save checkpoint
    checkpoint_dir = os.path.join(dir, f"checkpoint_epoch{epoch}_step{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(dir)

    # save the global min 
    if loss.item() < global_min:
        checkpoint_dir = os.path.join(dir, f"global_min")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(dir)
        global_min = loss.item()
    
    return global_min