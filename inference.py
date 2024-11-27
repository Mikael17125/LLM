import torch
from models.gpt import GPT
import tiktoken
import os

torch.manual_seed(123)

def load_checkpoint(model, filename="GPT2_Model.pth"):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {filename}")
    return model

def generate_text_simple(model, idx, max_new_tokens, context_size, device):
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def generate(model, idx, max_new_tokens, context_size, device,
             temperature=0.0, top_k=None, eos_id=None):
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 768
    num_heads = 12
    num_layers = 10
    vocab_size = 50257
    context_length = 512

    model = GPT(embed_dim,
                num_heads,
                num_layers,
                vocab_size,
                context_length)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters:{total_params:,}")

    checkpoint_file = "checkpoint/GPT2_TinyStory.pth"
    if os.path.exists(checkpoint_file):
        model = load_checkpoint(model, checkpoint_file)

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "A boy named Tom wanted to be a superhero. He put on a cape and a mask and ran around the house. He pretended to fight bad guys and save people. "
    model.eval()

    out = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=40,
        context_size=context_length,
        device=device,
        top_k=25,
        temperature=1
    )

    out_simple = generate_text_simple(model, 
                                      idx=text_to_token_ids(start_context, tokenizer),
                                      max_new_tokens=40,
                                      context_size=context_length, 
                                      device= device)

    decode = token_ids_to_text(out, tokenizer)
    print(decode)

    print("-"*50)

    decode_simple = token_ids_to_text(out_simple, tokenizer)
    print(decode_simple)

if __name__ == "__main__":
    main()
