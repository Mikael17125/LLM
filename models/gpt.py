import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, embed_dim):
        super(FFN, self).__init__()

        self.linear_one = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.ReLU()
        self.linear_two = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.linear_two(self.gelu(self.linear_one(x))))
        return x

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super(MHA, self).__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        _, seq_len, _ = x.shape

        if self.context_length >= seq_len:
            attn_mask = self.mask[:seq_len, :seq_len]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]
        
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        output = self.proj(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, context_length):
        super(TransformerBlock, self).__init__()

        self.multi_head_att = MHA(embed_dim, num_heads, context_length)
        self.feed_forward = FFN(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.multi_head_att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x

class GPT(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 num_layers, 
                 vocab_size,
                 context_length):
        
        super(GPT, self).__init__()

        self.token_embed = nn.Embedding(vocab_size, 
                                     embed_dim)
        self.pos_embed = nn.Embedding(context_length, 
                                    embed_dim)

        self.layers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, context_length) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        _, seq_len = x.size()

        token_embed = self.token_embed(x)
        pos_embed = self.pos_embed(torch.arange(seq_len, 
                                                device=x.device))
        x = token_embed + pos_embed

        x = self.layers(x)

        x = self.norm(x)
        x = self.fc(x)
        return x

    def generate(self,
                 model, 
                 idx, 
                 max_new_tokens, 
                 context_size, 
                 device,
                 temperature=0.0, 
                 top_k=None, 
                 eos_id=None):
        
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

if __name__ == "__main__":

    embed_dim = 768
    num_heads = 12
    feedforward_dim = 4 * embed_dim
    num_layers = 12
    vocab_size = 50257
    context_length = 1024

    model = GPT(embed_dim, 
                num_heads, 
                num_layers, 
                vocab_size,
                context_length)
    
    tgt = torch.randint(0, vocab_size, (2, 20))
    output = model(tgt)
    print(output.shape)