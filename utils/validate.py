import torch
from torch.nn import functional as F

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)

            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            total_loss += loss

        print(f"Validation Loss: {total_loss/len(val_loader):.4f}")

    model.train()
    return loss