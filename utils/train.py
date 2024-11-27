import torch
import time
from utils.validate import validate
from torch.nn import functional as F

def save_checkpoint(model, optimizer, epoch, filename="checkpoint/GPT2_TinyStory.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def load_checkpoint(model, optimizer, filename="GPT2_Model.pth"):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {filename}, starting at epoch {start_epoch}")
    return model, optimizer, start_epoch

def train(model, train_loader, val_loader, optimizer, lr_scheduler, device, num_epochs, start_epoch=0):
    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):

        for input_batch, target_batch in train_loader:
            
            optimizer.zero_grad()

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)

            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

            loss.backward()
            optimizer.step()

            # Print progress from rank 0 only
            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}, Step {step}/{len(train_loader)}: Loss = {loss.item():.4f}, Time elapsed = {elapsed_time:.2f} sec")

            step += 1

            # Perform validation
            if step % 100 == 0:
                validate(model, val_loader, device)

            # Update learning rate scheduler
            lr_scheduler.step()

            if step % 100 == 0:
                save_checkpoint(model, optimizer, epoch)