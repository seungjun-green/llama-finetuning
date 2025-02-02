import torch
import torch.nn as nn
from data.squad_data import create_squad_dataloader
from models.lora import load_lora_applied_model
from tqdm import tqdm  


# this is for the squad dataste
def eval_loss(model_name, checkpoint_path, dev_file_path, batch_size, max_length, use_fp16=False):
    tokenizer, model = load_lora_applied_model(model_name, checkpoint_path, rank=8, alpha=16.0)
    tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    dev_dataloader = create_squad_dataloader(
        dev_file_path, tokenizer, batch_size, max_length
    )

    total_loss = 0.0
    total_steps = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Evaluating", unit="batch"):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    return avg_loss