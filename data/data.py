from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length):
        self.inputs = tokenizer(
            input_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        ).input_ids
        self.labels = tokenizer(
            target_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        ).input_ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def create_dataloader(input_texts, target_texts, tokenizer, batch_size, max_length):
    dataset = TextDataset(input_texts, target_texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size)
