import os
import torch
import hashlib
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TinyStoryDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, cache_dir='/data/nias/LLM/cache'):
        self.cache_dir = cache_dir
        self.input_ids = []
        self.target_ids = []
        
        # Generate a cache file name based on the text content
        cache_key = hashlib.md5(txt.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}_{max_length}_{stride}.pt")
        
        # If the cache file exists, load it; otherwise, tokenize and process the text
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            cached_data = torch.load(cache_file)
            self.input_ids = cached_data['input_ids']
            self.target_ids = cached_data['target_ids']
        else:
            print("Processing and caching data...")
            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            
            # Process chunks sequentially and track progress with tqdm
            for idx in tqdm(range(0, len(token_ids) - max_length, stride), desc="Processing data", unit="chunk"):
                input_chunk = token_ids[idx:idx + max_length]
                target_chunk = token_ids[idx + 1: idx + max_length + 1]
                
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
            
            # Save the processed data to the cache file
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save({
                'input_ids': self.input_ids,
                'target_ids': self.target_ids
            }, cache_file)
            print(f"Data cached at: {cache_file}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def tiny_story_dataloader(txt, tokenizer, batch_size=4, max_length=256,
                          stride=128, shuffle=True, drop_last=True, cache_dir='./cache', num_workers=4):
    dataset = TinyStoryDataset(txt, tokenizer, max_length, stride, cache_dir)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True, num_workers=num_workers)

    return dataloader