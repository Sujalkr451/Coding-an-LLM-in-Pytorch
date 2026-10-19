import torch
import torch.nn as nn 

#implementing self attention with no trainable weights

class SelfAttention(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_keys=nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
        self.W_values=nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
        self.W_queries=nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)

    def forward(self,x):
        keys=x @ self.W_keys.T 
        values=x @ self.W_values.T 
        queries=x @ self.W_queries.T 

        attention_scores= ((queries @ keys.T)/(keys.shape[-1])**0.5)
        attention_weights= torch.softmax(attention_scores, dim=-1)
        context_vectors= attention_weights @ values

        return context_vectors

