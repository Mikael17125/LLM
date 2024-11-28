from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from torch import nn

class Llama(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.llama = base_model
        self.lm_head = nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits
