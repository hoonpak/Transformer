import torch
from torch import nn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, d_k, d_v, is_masked):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.WO = nn.Linear(head*d_v, d_model, bias=False)
        
        self.scaler = torch.sqrt(torch.tensor(d_k))
        self.is_masked = is_masked
        self.head = head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
    def forward(self, Query, Key, Value, masked_info):
        """
        **INPUT SHAPE**
        masked_info - N, QL, L -> padding = True, else = False
        """
        QN, QL, QD = Query.shape
        KN, KL, KD = Key.shape
        VN, VL, VD = Value.shape
        
        query = self.WQ(Query).reshape(QN,QL,self.head,self.d_k).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        key = self.WK(Key).reshape(KN,KL,self.head,self.d_k).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        value = self.WV(Value).reshape(VN,VL,self.head,self.d_v).transpose(1,2) #N, L, D => N, L, H, D/H => N, H, L, D/H
        
        scaled_output = torch.matmul(query, key.transpose(-1,-2))/self.scaler #N, H, QL, D/H * N, H, D/H, L => N, H, QL, L
        
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
        # masked_info = masked_info.unsqueeze(1).repeat(1,self.head,1,1) #N, QL, L => N, H, QL, L
        masked_info = masked_info.unsqueeze(1) #N, 1, QL, L
        
        attn_score = scaled_output.masked_fill(masked_info, float("-inf")).softmax(-1).nan_to_num(0) #N, H, QL, L
        # attn_score = scaled_output.masked_fill(masked_info, -1e9).softmax(-1) #N, H, QL, L
        multi_outputs = torch.matmul(attn_score, value).transpose(1,2).reshape(VN, QL, self.d_model)  #N, H, QL, L * N, H, L, D/H => N, H, QL, D/H => N, QL, H, D/H => N, QL, D
        output = self.WO(multi_outputs) #N, QL, d_m
        return output
        
    def initialization(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.gain_a = nn.Parameter(torch.ones(d_model))
        self.gain_b = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-6
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.gain_a*(x-mean)/(std+self.eps)+self.gain_b #N, L, d_m
        return output

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
