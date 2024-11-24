import torch
from models.model import GPT
import tiktoken

torch.manual_seed(123)

def generate_text_simple(model, 
                         idx,
                         max_new_tokens, 
                         context_size): 
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

if __name__  == "__main__":
    embed_dim = 768
    num_heads = 12
    feed_forward_dim = 4*embed_dim
    num_layers = 12
    vocab_size = 50257
    
    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"

    encoded_tensor = text_to_token_ids(start_context, tokenizer)

    model = GPT(embed_dim, 
                num_heads, 
                feed_forward_dim, 
                num_layers, 
                vocab_size)
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    model.eval()

    out = generate_text_simple(model=model,
                               idx=encoded_tensor, 
                               max_new_tokens=6, 
                               context_size=256)

    decode = token_ids_to_text(out, tokenizer)
    print(decode)