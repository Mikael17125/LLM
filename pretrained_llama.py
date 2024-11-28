import torch
import tiktoken
from utils.train import train
from datasets import load_dataset
from dataset.tiny_story_loader import tiny_story_dataloader
import torch.distributed as dist
import os
import time
from transformers import LlamaModel, LlamaConfig
from models.llama import Llama

# Configuration
configuration = LlamaConfig(
    vocab_size=50257,  # Adjust based on tokenizer
    hidden_size=512,  # Default size for Llama
    num_attention_heads=4,  # Default for Llama
    num_hidden_layers=4,  # Number of transformer blocks
    intermediate_size= 4 * 512,  # Feed-forward layer size
    max_position_embeddings= 128,  # Adjust as needed
)

def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank) 
    return local_rank

def load_checkpoint(model, 
                    optimizer, 
                    filename="Llama.pth"):
    
    checkpoint = torch.load(filename, 
                            map_location='cpu')
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded from {filename}, starting at epoch {start_epoch}")

    return model, optimizer, start_epoch

def main():
    local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    num_epochs = 1

    tokenizer = tiktoken.get_encoding("gpt2")
    base_model = LlamaModel(configuration)

    model = Llama(base_model)
    
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[local_rank])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    data_start = time.time()
    dataset = load_dataset('roneneldan/TinyStories')
    text = '\n'.join(dataset['train']['text'][:])
    n = int(0.9*len(text))
    print(f"Load Dataset: {time.time() - data_start}")

    train_data = text[:n]
    val_data = text[n:]

    train_loader = tiny_story_dataloader(
        train_data,
        batch_size=128,
        tokenizer=tokenizer,
        max_length=configuration.max_position_embeddings,
        stride=configuration.max_position_embeddings,
        drop_last=True,
        num_workers=4,
    )

    val_loader = tiny_story_dataloader(
        val_data,
        batch_size=128,
        tokenizer=tokenizer,
        max_length=configuration.max_position_embeddings,
        stride=configuration.max_position_embeddings,
        drop_last=False,
        num_workers=4,
        shuffle=False
    )

    print(f"Load Dataset Loader: {time.time() - data_start}")

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=3e-4, 
                                  weight_decay=0.1)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                              T_max=num_epochs)

    checkpoint_file = "_"
    if os.path.exists(checkpoint_file):
        model, optimizer, start_epoch = load_checkpoint(model, 
                                                        optimizer, 
                                                        checkpoint_file)
    else:
        start_epoch = 0

    save_step = 1000 

    train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          lr_scheduler, 
          device, 
          num_epochs, 
          start_epoch,
          save_step)

if __name__ == "__main__":
    main()