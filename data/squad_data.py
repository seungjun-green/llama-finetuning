from torch.utils.data import DataLoader, Dataset

import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class SQuADDataset(Dataset):
    def __init__(self, context_qa_map, tokenizer, max_length):
        """
        Custom Dataset for SQuAD-like data.

        Args:
            context_qa_map (dict): Dictionary mapping context to a list of (question, answer) pairs.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode input and target texts.
            max_length (int): Maximum token length for inputs and labels.
        """
        self.data = []
        for context, qa_pairs in context_qa_map.items():
            for question, answer in qa_pairs:
                self.data.append((context, question, answer))
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data item as input and label tensors.

        Args:
            idx (int): Index of the data item.

        Returns:
            inputs (torch.Tensor): Encoded input text tensor.
            labels (torch.Tensor): Encoded target text tensor.
        """
        context, question, answer = self.data[idx]
        input_text = f"context: {context} question: {question}"
        target_text = answer

        inputs = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        ).input_ids.squeeze(0)
        labels = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        ).input_ids.squeeze(0)

        return inputs, labels
    

def extract_squad_data_optimized(file_path):
    """
    Extract data from SQuAD-like JSON file and structure it by context.

    Args:
        file_path (str): Path to the SQuAD JSON file.

    Returns:
        dict: Dictionary mapping context to a list of (question, answer) pairs.
    """
    with open(file_path, 'r') as file:
        squad_data = json.load(file)
    
    context_qa_map = defaultdict(list)

    for entry in squad_data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text'] if qa['answers'] else ""
                context_qa_map[context].append((question, answer))
    
    return context_qa_map

def create_squad_dataloader(file_path, tokenizer, batch_size, max_length):
    """
    Create a DataLoader for SQuAD-like data.

    Args:
        file_path (str): Path to the SQuAD JSON file.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum token length for inputs and labels.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    context_qa_map = extract_squad_data_optimized(file_path)
    dataset = SQuADDataset(context_qa_map, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)