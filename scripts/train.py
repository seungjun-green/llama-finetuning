import os
from torch.optim import Adam
from transformers import get_scheduler
from configs.squad_config import SquadFineTuneConfig
from models.base_model import load_base_model
from models.lora import add_lora_to_model
from utils.helpers import count_params
from models.loss import get_loss_function
from data.squad_data import create_squad_dataloader
from tqdm import tqdm

def fine_tune(config_filepath, **kwargs):
    global_min = 10
    # load config
    config = SquadFineTuneConfig(config_path=config_filepath, **kwargs)
    
    total_training_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(0.1 * total_training_steps)  
    

    # load tokenizer and model
    tokenizer, model = load_base_model(config.base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # add LoRA layers to the model
    model = add_lora_to_model(model, rank=config.lora_rank, alpha=config.lora_alpha)

    # unfreeze lora parameters
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name

    # initalize optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    # Print trainable parameters
    total_params, trainable_params = count_params(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    loss_fn = get_loss_function(loss_type="cross_entropy")
    
    # work on dataloader

    # Fine-tuning loop
    model.train()
    
    train_dataloader = create_squad_dataloader(config.train_file_path, tokenizer, config.batch_size, config.max_length)
    # dev_dataloader = create_squad_dataloader(config.dev_file_path, tokenizer, config.batch_size, config.max_length)
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, (input_ids, labels) in progress_bar:
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits # (N, seq_length, vocab_size)
            # logits.view(-1, logits.size(-1)): (N*seq_length, vocab_size)
            # labels.view(-1): (N*seq_length,)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"Step": step + 1, "Loss": loss.item()})

            if step % config.log_steps == 0:
                print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
                # save checkpoint
                checkpoint_dir = os.path.join(config.output_dir, f"checkpoint_epoch{epoch}_step{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(config.output_dir)
                
                # save the global min 
                if loss.item() < global_min:
                    checkpoint_dir = os.path.join(config.output_dir, f"global_min")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(config.output_dir)
                    global_min = loss.item()
