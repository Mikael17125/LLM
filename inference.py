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
    num_heads = 4
    num_layers = 2
    vocab_size = 50257
    context_length = 128

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

    start_context = "Once upon a time there was"
    model.eval()

    token_ids = model.generate(model=model,
                                idx=text_to_token_ids(start_context, tokenizer),
                                max_new_tokens=512,
                                context_size=context_length,
                                device=device,
                                top_k=25,
                                temperature=1.5)

    decode = token_ids_to_text(token_ids, tokenizer)
    print(decode)

if __name__ == "__main__":
    main()
