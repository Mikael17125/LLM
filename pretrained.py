import torch
from models.gpt import GPT
import tiktoken
from utils.train import train
from datasets import load_dataset
from dataset.tiny_story_loader import tiny_story_dataloader
import torch.distributed as dist
import os
import time

def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank) 
    return local_rank

def load_checkpoint(model, 
                    optimizer, 
                    filename="GPT2_Model.pth"):
    
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

    embed_dim = 768
    num_heads = 4
    num_layers = 2
    vocab_size = 50257
    context_length = 128
    num_epochs = 1

    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(embed_dim, 
                num_heads, 
                num_layers, 
                vocab_size, 
                context_length)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

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
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        num_workers=4,
    )

    val_loader = tiny_story_dataloader(
        val_data,
        batch_size=128,
        tokenizer=tokenizer,
        max_length=context_length,
        stride=context_length,
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

    save_step = 500

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