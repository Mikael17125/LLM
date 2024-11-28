import torch
from models.gpt import GPT
from transformers import LlamaModel, LlamaConfig
from models.llama import Llama
import tiktoken
import os

torch.manual_seed(123)

# Configuration
configuration = LlamaConfig(
    vocab_size=50257,  # Adjust based on tokenizer
    hidden_size=512,  # Default size for Llama
    num_attention_heads=4,  # Default for Llama
    num_hidden_layers=4,  # Number of transformer blocks
    intermediate_size= 4 * 512,  # Feed-forward layer size
    max_position_embeddings= 128,  # Adjust as needed
)

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
    base_model = LlamaModel(configuration)

    model = Llama(base_model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters:{total_params:,}")

    checkpoint_file = "/media/user/Data/Code/LLM/checkpoint/Llama_TinyStory.pth"
    if os.path.exists(checkpoint_file):
        model = load_checkpoint(model, checkpoint_file)

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Once upon a time there was"
    model.eval()

    out = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=512,
        context_size= configuration.max_position_embeddings,
        device=device,
        top_k=25,
        temperature=5
    )

    decode = token_ids_to_text(out, tokenizer)
    print(decode)

if __name__ == "__main__":
    main()
