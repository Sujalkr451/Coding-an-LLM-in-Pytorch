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
        keys=x @ self.W_keys
        values=x @ self.W_values
        queries=x @ self.W_queries

        attention_scores= ((queries @ keys.T)/(keys.shape[-1])**0.5)
        attention_weights= torch.softmax(attention_scores, dim=-1)
        context_vectors= attention_weights @ values

        return context_vectors
"""
nn.parameter layer isn't efficient as nn.linear layer in pytorch so we implement self attention class with nn.linear layers 

"""


class SelfAttentionV(nn.Module):
    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()

        self.W_query=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_keys=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_values=nn.Linear(d_in,d_out, bias=qkv_bias)

    def forward(self,x):
        keys = self.W_keys(x)
        values=self.W_values(x)
        queries=self.W_query(x)

        attention_score= queries@keys.T
        attention_weights= torch.softmax(attention_score/queries.shape[-1]**0.5, dim=-1)

        context_vectors=attention_weights@values

        return context_vectors

