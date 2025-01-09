# ARTICLEGENERATOR/scripts/train.py

import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_scheduler
from configs.config import FineTuneConfig
from configs.config import Config
from models.base_model import load_base_model
from models.lora import add_lora_to_model
from utils.helpers import count_params
from models.loss import get_loss_function


def fine_tune(config_filepath):
    # load config
    config = Config(config_path=config_filepath)

    # load tokenizer and model
    tokenizer, model = load_base_model(config.base_model_name)

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

    # Sample dataset
    input_texts = ["Hello, how are you?", "This is a fine-tuning example."]
    target_texts = ["I'm good, thank you!", "Indeed, it is!"]

    # Tokenize and create dataloader
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt").input_ids
    dataset = list(zip(inputs.input_ids, labels))
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    loss_fn = get_loss_function(loss_type="cross_entropy")

    # Fine-tuning loop
    model.train()
    for epoch in range(config.num_epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits # (N, seq_length, vocab_size)
            # logits.view(-1, logits.size(-1)): (N*seq_length, vocab_size)
            # labels.view(-1): (N*seq_length,)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % config.save_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                # Save checkpoint
                os.makedirs(config.output_dir, exist_ok=True)
                model.save_pretrained(config.output_dir)
                tokenizer.save_pretrained(config.output_dir)
