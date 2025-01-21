from torch.optim import Adam
from transformers import get_scheduler
from configs.squad_config import SquadFineTuneConfig
from models.lora import add_lora_to_model
from utils.helpers import count_params
from data.squad_data import create_squad_dataloader
from utils.checkpoint import save_checkponit
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.optim import AdamW

def fine_tune(model, tokenizer, config_filepath, **kwargs):
    global_min = 9
    
    config = SquadFineTuneConfig(config_path=config_filepath, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    use_fp16 = getattr(config, "use_fp16", False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # add LoRA layers to the model
    model = add_lora_to_model(model, rank=config.lora_rank, alpha=config.lora_alpha)
    model.to(device)

    # unfreeze lora parameters
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name

    # Print trainable parameters
    total_params, trainable_params = count_params(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    scaler = GradScaler() if use_fp16 else None
    
    # Fine-tuning loop
    model.train()
    
    train_dataloader = create_squad_dataloader(config.train_file_path, tokenizer, config.batch_size, config.max_length)
        
    total_training_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(config.warmup_ratio * total_training_steps)  
    
    # initalize optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),            # Default betas for AdamW
        weight_decay=0.01              # Recommended weight decay
    )
    
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_training_steps
    )

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if use_fp16:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() 
            
            else:
                outputs = model(inputs)
                '''
                outputs.view(-1, logits.size(-1)): (N*seq_length, vocab_size)
                labels.view(-1): (N*seq_length,)
                '''
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"Step": step + 1, "Loss": loss.item()})
            global_min = save_checkponit(model, config.output_dir, epoch, step, loss, global_min, config.log_steps, len(train_dataloader))
