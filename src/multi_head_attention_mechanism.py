import torch 
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,num_heads, dropout=0.2,qkv_bias=False ):
        super().__init__()
        assert (d_out % num_heads == 0)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 

        self.w_query=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.w_keys=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.w_values=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.Dropout=nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length),
                               diagonal=1))

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        query= self.w_query(x)
        keys=self.w_keys(x)
        values=self.w_values(x)

        query=query.view(b,num_tokens, self.num_heads, self.head_dim)
        keys=keys.view(b,num_tokens, self.num_heads, self.head_dim)
        values=values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        query=query.transpose(1,2)
        values=values.transpose(1,2)

        attention_score= query @ keys.transpose(2,3)

        mask_bool= self.mask.bool()[:num_tokens, :num_tokens]
        attention_score.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attention_score / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) # we want the original dimension back for embedding vectors



        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) #contiguous to make sure they are in same memory block
        
        context_vec = self.out_proj(context_vec) # optional 
        
        return context_vec


