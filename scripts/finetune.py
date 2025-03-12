import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from tqdm import tqdm
from utils.helpers import count_params
from data.json_data import create_dataloaders
from configs.finetune_config import FineTuneConfig
from models.base_model import load_base_model
from models.lora import LoRALinear
from models.lora import add_lora_to_model
from models.dora import DoRALinear
from models.dora import add_dora_to_model
class Finetuner:
    def __init__(self, finetune_method, config_filepath, **kwargs):
        self.config = FineTuneConfig(config_path=config_filepath, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.config.base_model_name
        
        # early stopping
        self.best_val_loss = float('inf') 
        self.early_stopping_patience = self.config.patience
        self.no_improvement_count = 0  
        
        # load tokenizer and model
        self.tokenizer, self.model = load_base_model(self.model_name)
        self.model = self.model.to(self.device)
        
        # fp16 setting
        self.use_fp16 = getattr(self.config, "use_fp16", False)
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # set the pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        
        # add LoRA layers to the model
        # add option here[lora or dora or something else]
        if finetune_method == "lora":
            self.model = add_lora_to_model(self.model, rank=self.config.lora_rank, alpha=self.config.lora_alpha)
        elif finetune_method == "dora":
            self.model = add_dora_to_model(self.model)
        else:
            raise ValueError(f"UnSupported fine tuning method: {finetune_method}")
        
        self.model.to(self.device)
        
        for name, param in self.model.named_parameters():
            if finetune_method == "lora":
                param.requires_grad = "lora" in name
            elif finetune_method == "dora":
                param.requires_grad = "dora" in name
                
        # Print trainable parameters
        total_params, trainable_params = count_params(self.model)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Create train dataLoader and val dataloader
        self.train_dataloader, self.val_loader = create_dataloaders(
            self.config.train_file_path, 
            self.tokenizer, 
            self.config.batch_size, 
            self.config.max_length,
            self.config.train_ratio
        )
        
        # Optimizer and scheduler setup
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        total_training_steps = len(self.train_dataloader) * self.config.num_epochs
        warmup_steps = int(self.config.warmup_ratio * total_training_steps)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        
        
        self.log = self.config.log_steps
        
        self.train_losses = []
        self.val_losses = []
        
    def save_lora_weights(self, model, save_directory, check_name):
        os.makedirs(save_directory, exist_ok=True)
        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_a.weight"] = module.lora_a.weight.detach().cpu()
                lora_state_dict[f"{name}.lora_b.weight"] = module.lora_b.weight.detach().cpu()

        torch.save(lora_state_dict, os.path.join(save_directory, check_name))
        
    
    def save_dora_weights(self, model, save_directory, check_name):
        os.makedirs(save_directory, exist_ok=True)
        dora_state_dict = {}
        
        for name, module in model.named_modules():
            if isinstance(module, DoRALinear):
                # Save trainable DoRA parameters
                dora_state_dict[f"{name}.dora_m"] = module.m.detach().cpu()
                dora_state_dict[f"{name}.dora_B"] = module.B.detach().cpu()
                dora_state_dict[f"{name}.dora_A"] = module.A.detach().cpu()
        
        # Save the state dictionary to a file
        save_path = os.path.join(save_directory, check_name)
        torch.save(dora_state_dict, save_path)
        print(f"[INFO] DoRA weights saved to {save_path}")
        
    def get_val_loss(self):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, labels in self.val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    
                logits = outputs.logits
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()
                num_batches += 1
            
        avg_val_loss = total_val_loss / num_batches
        self.model.train() 
        return avg_val_loss
        
    def train(self):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}")
            for step, (input_ids, labels) in progress_bar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        logits = outputs.logits
                        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    logits = outputs.logits
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()
                    self.optimizer.step()
                
                self.lr_scheduler.step()
                self.train_losses.append(loss)
                
                progress_bar.set_postfix({"Step": step + 1, "Loss": loss.item()})
                
                # Save checkpoint based on some logging steps and improvement criteria
                if  step % self.log == 0 and step != 0 or step == len(self.train_dataloader) - 1:
                    # get the validation loss
                    val_loss = self.get_val_loss()
                    self.val_losses.append(val_loss)
                    tqdm.write(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {round(val_loss, 4)}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.no_improvement_count = 0
                        # save the current checkpoint
                        if self.finetune_method == "lora":
                            self.save_lora_weights(self.model, self.config.output_dir, f"epoch{epoch}_step{step}_loss{round(val_loss, 4)}")
                        elif self.finetune_method == "dora":
                            self.save_dora_weights(self.model, self.config.output_dir, f"epoch{epoch}_step{step}_loss{round(val_loss, 4)}")
                        else:
                            print("Bro, what the hack ru doing? r u nuts. smh.")
                            
                    else:
                        self.no_improvement_count += 1
                        
                        if self.no_improvement_count >= self.early_stopping_patience:
                            tqdm.write(f"Early stopping triggered after {self.early_stopping_patience} consecutive logs without improvement.")
                            return
                            