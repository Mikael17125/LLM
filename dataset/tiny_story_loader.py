import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist


class TinyStoryDataset(Dataset):
    def __init__(self, 
                 txt, 
                 tokenizer, 
                 max_length, 
                 stride):
        
        self.input_ids = []
        self.target_ids = []
        
        print("Tokenizing...")
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for idx in tqdm(range(0, len(token_ids) - max_length, stride), desc="Processing data", unit="chunk"):
            input_chunk = token_ids[idx:idx + max_length]
            target_chunk = token_ids[idx + 1: idx + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def tiny_story_dataloader(txt, 
                          tokenizer, 
                          batch_size=4, 
                          max_length=256,
                          stride=128, 
                          drop_last=True, 
                          num_workers=4,
                          shuffle=True):
    
    dataset = TinyStoryDataset(txt, 
                               tokenizer, 
                               max_length, 
                               stride)
                               
    sampler = DistributedSampler(dataset, 
                                 shuffle=shuffle,
                                 num_replicas=dist.get_world_size(), rank=dist.get_rank())

    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size,
        drop_last=drop_last, 
        pin_memory=True, 
        num_workers=num_workers)

    return dataloader
