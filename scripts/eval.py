import torch
import torch.nn as nn
from data.squad_data import create_squad_dataloader


def construct_model(model, lora_weigt):
    pass

def eval_loss(model, tokenizer, dev_file_path, batch_size, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dev_dataloader = create_squad_dataloader(dev_file_path, tokenizer, batch_size, max_length)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for input_ids, labels in dev_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits  # (N, seq_length, vocab_size)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0

def eval_em(model, tokenizer, dev_file_path, batch_size, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dev_dataloader = create_squad_dataloader(dev_file_path, tokenizer, batch_size, max_length)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for input_ids, labels in dev_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits  # (N, seq_length, vocab_size)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0