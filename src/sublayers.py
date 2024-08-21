import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, is_masked):
        super(ScaleDotProductAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k, bias=False)
        self.WK = nn.Linear(d_model, d_k, bias=False)
        self.WV = nn.Linear(d_model, d_v, bias=False)
        self.scaler = torch.sqrt(torch.tensor(d_k))
        self.is_masked = is_masked
        
    def forward(self, Q, K, V, masked_info):
        """
        **INPUT SHAPE**
        masked_info - N, L, L -> padding = True, else = False
        """
        query = self.WQ(Q) #N, L, d_k
        key = self.WK(K) #N, L, d_k
        value = self.WV(V) #N, L, d_v
        scaled_output = torch.bmm(query, key.permute(0,2,1))/self.scaler #N, L, L
        
        # masked_info = (torch.bmm(masked_info.unsqueeze(2), masked_info.unsqueeze(1)) == 0) #padding masking / N, L, L
        
        if self.is_masked: #auto-regressive masking
            """
            [[False,  True,  True,  True,  True],
             [False, False,  True,  True,  True],
             [False, False, False,  True,  True],
             [ True,  True,  True,  True,  True],
             [ True,  True,  True,  True,  True]]
            """
            leftward_mask = (torch.triu(torch.ones_like(masked_info), 1) != 0)
            masked_info = masked_info | leftward_mask
        
        attn_score = scaled_output.masked_fill_(masked_info, float("-inf")).softmax(-1).nan_to_num(0) #N, L, L
        output = torch.bmm(attn_score, value) #N, L, d_v
        return output

    def initialization(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, d_k, d_v, is_masked):
        super(MultiHeadAttention, self).__init__()
        self.attention_layers = nn.ModuleList([ScaleDotProductAttention(d_model, d_k, d_v, is_masked) for h in range(head)])
        self.WO = nn.Linear(head*d_v, d_model, bias=False)
        
    def forward(self, Q, K, V, masked_info):
        head_outputs = []
        for attn_layer in self.attention_layers:
            head_out = attn_layer(Q, K, V, masked_info)
            head_outputs.append(head_out)
        multi_outputs = torch.cat(head_outputs, dim=-1) #N, L, d_v*h
        output = self.WO(multi_outputs) #N, L, d_m
        return output
        
    def initialization(self):
        for attn_layer in self.attention_layers:
            attn_layer.initialization()
        nn.init.xavier_uniform_(self.WO.weight)
        
class LayerLorm(nn.Module):
    def __init__(self, d_model):
        super(LayerLorm, self).__init__()
        self.gain = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.gain(((x-mean)/(std+1e-6))) #N, L, d_m
        return output
        
    def initialization(self):
        nn.init.ones_(self.gain.weight)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.inner_layer = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.outer_layer = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """
        **INPUT SHAPE**
        x -> N, L, d_model
        """
        inner_output = self.relu(self.inner_layer(x))
        outer_output = self.outer_layer(inner_output) #N, L, d_model
        return outer_output
    
    def initialization(self):
        nn.init.xavier_uniform_(self.inner_layer.weight)
        nn.init.xavier_uniform_(self.outer_layer.weight)
