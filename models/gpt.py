import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()

        self.linear_one = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.linear_two = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear_two(self.gelu(self.linear_one(x)))
        return x
    
class MultiHeadAtt(nn.Module):
    def __init__(self, embed_dim, num_heads, context_length):
       super(MultiHeadAtt, self).__init__()

       self.embed_dim = embed_dim
       self.num_head = num_heads
       self.head_dim = embed_dim // num_heads
       assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
       
       self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
       self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
       self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
       
       self.out_proj = nn.Linear(embed_dim, embed_dim)
       self.softmax = nn.Softmax(dim=-1)
       self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-1,-2)) /  self.head_dim ** 0.5
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        
        scores = scores.masked_fill_(mask_bool == 0, -1e9)

        attention_weights = self.softmax(scores)
        attention_outputs = torch.matmul(attention_weights, V)

        attention_outputs = attention_outputs.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attention_outputs)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, context_length):
        super(TransformerBlock, self).__init__()

        self.multi_head_att = MultiHeadAtt(embed_dim, num_heads, context_length)
        self.feed_forward = FeedForward(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.multi_head_att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.multi_head_att(x)
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

        
        self.dropout_embed = nn.Dropout(0.1)

        self.layers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, context_length) for _ in range(num_layers)])
        

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        _, seq_len = x.shape
        
        token_embed = self.token_embed(x)
        pos_embed = self.pos_embed(torch.arange(seq_len, device=x.device))
        x = token_embed + pos_embed 
        x = self.dropout_embed(x)
        x = self.layers(x)

        x = self.norm(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":

    embed_dim = 768
    num_heads = 12
    feedforward_dim = 4 * embed_dim
    num_layers = 12
    vocab_size = 50257
    context_length = 1024

    model = GPT(embed_dim, 
                num_heads, 
                feedforward_dim, 
                num_layers, 
                vocab_size,
                context_length)
    
    
    tgt = torch.randint(0, vocab_size, (2, 20))
    output = model(tgt)
    print(output.shape)