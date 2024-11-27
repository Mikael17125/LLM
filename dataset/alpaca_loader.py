from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import torch
from functools import partial


class TinyStoryDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.tokenizer = tokenizer

        # Load JSON file
        try:
            with open(filename, 'r') as file:
                self.dataset = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON file '{filename}'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text = data['text']

        # Tokenize and limit length
        chunk = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
       
        return chunk

def tiny_story_custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])      
        targets = torch.tensor(padded[1:])     

        mask = targets == pad_token_id             
        indices = torch.nonzero(mask).squeeze()    
        if indices.numel() > 1:                     
            targets[indices[1:]] = ignore_index   

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]      
            targets = targets[:allowed_max_length]    

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    filename = "samples/tinyStories/validation.json"
    max_len = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(
        tiny_story_custom_collate_fn,
        device=device,
        allowed_max_length=max_len
    )
    try:
        tiny_story_dataset = TinyStoryDataset(filename, tokenizer)
        dataloader = DataLoader(
            tiny_story_dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=customized_collate_fn
        )

        # Test data loading
        for batch in dataloader:
            inputs, targets = batch
            print(f"Inputs: {inputs}, Targets: {targets}")
            break
    except Exception as e:
        print(f"Error: {e}")