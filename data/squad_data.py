from torch.utils.data import DataLoader, Dataset
import torch
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class SQuADDataset(Dataset):
    def __init__(self, context_qa_map, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for context, qa_pairs in context_qa_map.items():
            for question, answer in qa_pairs:
                input_text = f"context: {context}\nquestion: {question}\nanswer:"
                target_text = answer

                # Tokenize and apply truncation here
                # first tokenize both input_text and target_text
                # input_text: remove the 128001 in the end
                # output_test: remove the 12800 in the start
                input_tensor = list(tokenizer.encode(input_text)[:-1])  
                target_tensor = list(tokenizer.encode(target_text)[1:])
                
                # truncate
                if len(input_tensor) > self.max_length:
                    input_tensor = input_tensor[:self.max_length]

                # Step2: do the padding process
                # input_tensor: add X number of 128004 to the right, where X = max_seq_length - current length
                # target_tensor: add 'origianl number of input_tensor' number of 128002 in the left, and 128004 to the right
                input_tensor_length = len(input_tensor)
                target_tensor_length = len(target_tensor)
                
                input_right_pad_nums = max(0, self.max_length - input_tensor_length)
                target_right_pad_nums = max(0, self.max_length - (input_tensor_length + target_tensor_length))

                input_tensor += [128004] * input_right_pad_nums  # Pad right
                target_tensor = [128004] * input_tensor_length + target_tensor + [128004] * target_right_pad_nums
                
                if len(target_tensor) > self.max_length:
                    target_tensor = target_tensor[:self.max_length]
                    
                self.data.append((torch.LongTensor(input_tensor), torch.LongTensor(target_tensor)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
    

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