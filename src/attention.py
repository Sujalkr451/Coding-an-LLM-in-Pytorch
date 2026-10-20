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


"""
Implementation of causal attention

"""


class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,dropout=0.2,qkv_bias=False):
        super().__init__()

        self.w_query=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.w_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.Dropout=nn.Dropout2d(dropout)

    def forward(self,x):
        query=self.w_query(x)
        key=self.w_key(x)
        value=self.w_value(x)

        context_length=x.shape[0] #number of token

        attention_score=query @ key.T

        mask= torch.triu(torch.ones(context_length, context_length), diagonal=1)

        masked= attention_score.masked_fill(mask.bool(), -torch.inf)

        attention_weight=torch.softmax(attention_score/key.shape[-1]**0.5, dim=-1)

        self.Dropout(masked)

        context_vector= masked@value

        return context_vector



