import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, is_decode):
        super(ScaleDotProductAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k, bias=False)
        self.WK = nn.Linear(d_model, d_k, bias=False)
        self.WV = nn.Linear(d_model, d_v, bias=False)
        self.scaler = torch.sqrt(torch.tensor(d_k))
        self.is_decode = is_decode
        
    def forward(self, x, masked_info):
        """
        **INPUT SHAPE**
        masked_info - N, L -> padding = 0, else = 1
        """
        query = self.WQ(x) #N, L, d_k
        key = self.WK(x) #N, L, d_k
        value = self.WV(x) #N, L, d_v
        scaled_output = torch.bmm(query, key.permute(0,2,1))/self.scaler #N, L, L
        
        masked_info = (torch.bmm(masked_info.unsqueeze(2), masked_info.unsqueeze(1)) == 0) #padding masking / N, L, L
        if self.is_decode: #auto-regressive masking
            leftward_mask = (torch.triu(torch.ones_like(masked_info), 1) != 0)
            masked_info = masked_info | leftward_mask
        
        attn_score = scaled_output.masked_fill_(masked_info, float("-inf")).softmax(-1).nan_to_num_(0) #N, L, L
        output = torch.bmm(attn_score, value) #N, L, d_v
        return output

    def initialization(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, d_k, d_v, is_decode):
        super(MultiHeadAttention, self).__init__()
        self.attention_layers = nn.ModuleList([ScaleDotProductAttention(d_model, d_k, d_v, is_decode) for h in range(head)])
        self.WO = nn.Linear(head*d_v, d_model, bias=False)
        
    def forward(self, x, masked_info):
        head_outputs = []
        for attn_layer in self.attention_layers:
            head_out = attn_layer(x, masked_info)
            head_outputs.append(head_out)
        multi_outputs = torch.cat(head_outputs, dim=-1) #N, L, d_v*h
        output = self.WO(multi_outputs) #N, L, d_m
        return output
        
    def initialization(self):
        for attn_layer in self.attention_layers:
            attn_layer.initialization()
        nn.init.xavier_uniform_(self.WO)
        
class LayerLorm(nn.Module):
    def __init__(self, d_model):
        super(LayerLorm, self).__init__()
        self.gain = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.gain(((x-mean)/(std+1e-6)))
        return output
        
    def initialization(self):
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

class FeedForward(nn.Module):
    def __init__(self, ):
        super(FeedForward, self).__init__()
        
    def forward(self, ):
        """"""