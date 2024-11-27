import torch
from models.gpt import GPT
import tiktoken
from utils.train import train
from datasets import load_dataset
from dataset.tiny_story_loader import tiny_story_dataloader
import os
import time

def load_checkpoint(model, optimizer, filename="GPT2_Model.pth"):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {filename}, starting at epoch {start_epoch}")
    return model, optimizer, start_epoch

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 768
    num_heads = 12
    num_layers = 10
    vocab_size = 50257
    context_length = 512
    num_epochs = 4

    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(embed_dim, num_heads, num_layers, vocab_size, context_length)
    model = model.to(device)

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
        batch_size=32,
        tokenizer=tokenizer,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

    val_loader = tiny_story_dataloader(
        val_data,
        batch_size=32,
        tokenizer=tokenizer,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=4
    )
    print(f"Load Dataset Loader: {time.time() - data_start}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
    checkpoint_file ="_"
    if os.path.exists(checkpoint_file):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)
    else:
        start_epoch = 0
    
    train(model, train_loader, val_loader, optimizer, lr_scheduler, device, num_epochs, start_epoch=start_epoch)

if __name__ == "__main__":
    main()