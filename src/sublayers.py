import torch
from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, is_decode) -> None:
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
        value = self.WV(x) #N, L, d_k
        scaled_output = torch.bmm(query, key.permute(0,2,1))/self.scaler #N, L, L
        masked_info = torch.bmm(masked_info.unsqueeze(2), masked_info.unsqueeze(1))
        if self.is_decode:
            leftward_mask = (torch.triu(torch.ones_like(masked_info), 1) != 0)
            masked_info = (masked_info == 0) | leftward_mask
        masked_output = scaled_output.masked_fill_(masked_info, float("-inf"))
        
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, ) -> None:
        super(MultiHeadAttention, self).__init__()
        
    def forward(self, ):
        """"""
        
class LayerLorm(nn.Module):
    def __init__(self, ) -> None:
        super(LayerLorm, self).__init__()
        
    def forward(self, ):
        """"""

class FeedForward(nn.Module):
    def __init__(self, ) -> None:
        super(FeedForward, self).__init__()
        
    def forward(self, ):
        """"""