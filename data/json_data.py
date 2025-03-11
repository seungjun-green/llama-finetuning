import json
from torch.utils.data import random_split, DataLoader, Dataset
import torch

class JSON_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        label_text = item['label']
        target_text = f"{input_text} {label_text} <|end_of_text|>"

        input_encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = input_encoding.input_ids.squeeze(0)
        attention_mask = input_encoding.attention_mask.squeeze(0)

        target_encoding = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = target_encoding.input_ids.squeeze(0)


        prompt_length = torch.sum(attention_mask).item()
        labels[:prompt_length] = 128004

        return input_ids, labels
    
def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_dataloaders(file_path, tokenizer, batch_size, max_length, train_ratio):
    data = load_data_from_json(file_path)
    dataset = JSON_Dataset(data, tokenizer, max_length)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader