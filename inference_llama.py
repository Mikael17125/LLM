import torch
from models.gpt import GPT
from transformers import LlamaModel, LlamaConfig
from models.llama import Llama
import tiktoken
import os

torch.manual_seed(123)

# Configuration
config = LlamaConfig(
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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Llama(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters:{total_params:,}")

    checkpoint_file = "checkpoint/Llama_TinyStory.pth"
    if os.path.exists(checkpoint_file):
        model = load_checkpoint(model, checkpoint_file)

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Once upon a time there was"
    model.eval()

    token_ids = model.generate(model=model,
                         idx=text_to_token_ids(start_context, tokenizer),
                         max_new_tokens=40,
                         context_size=config.max_position_embeddings,
                         device=device,
                         top_k=25,
                         temperature=1.5)

    decode = token_ids_to_text(token_ids, tokenizer)
    print(decode)

if __name__ == "__main__":
    main()