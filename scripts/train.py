import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_scheduler
from configs.config import FineTuneConfig
from configs.config import FineTuneConfig
from models.base_model import load_base_model
from models.lora import add_lora_to_model
from utils.helpers import count_params
from models.loss import get_loss_function
from data.data import create_dataloader


def fine_tune(config_filepath):
    # load config
    config = FineTuneConfig(config_path=config_filepath)

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
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=1000)

    # Print trainable parameters
    total_params, trainable_params = count_params(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    loss_fn = get_loss_function(loss_type="cross_entropy")
    
    # work on dataloader

    # Fine-tuning loop
    model.train()
    input_texts = [
        "Question: What is the capital of France? Context: France is a country in Europe. Its capital city is Paris.",
        "Question: Who wrote 'To Kill a Mockingbird'? Context: 'To Kill a Mockingbird' is a novel written by Harper Lee."
    ]
    
    target_texts = [
        "Paris",
        "Harper Lee"
    ]
    
    dataloader = create_dataloader(input_texts, target_texts, tokenizer, config.batch_size, config.max_seq_length)
    for epoch in range(config.num_epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            print(input_ids.shape)
            print(labels.shape)
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits # (N, seq_length, vocab_size)
            # logits.view(-1, logits.size(-1)): (N*seq_length, vocab_size)
            # labels.view(-1): (N*seq_length,)
            print(logits.shape)
            print(labels.shape)
            print(logits.view(-1, logits.size(-1)).shape)
            print(labels.view(-1).shape)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % config.save_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                # save checkpoint
                checkpoint_dir = os.path.join(config.output_dir, f"checkpoint_epoch{epoch}_step{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(config.output_dir)
                
