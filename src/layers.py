import torch
from torch import nn

import info

class EmbeddingWithPosition(nn.Module):
    def __init__(self, num_embeddings, pos_max_len, embedding_dim, shared_parameter) -> None:
        super(EmbeddingWithPosition, self).__init__()
        self.dim = torch.tensor(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=info.PAD)
        self.embedding.weight = shared_parameter
        self.pos_enc = self.get_pos_encoding(dim=embedding_dim, max_len=pos_max_len) #L, D
        
    def forward(self, x):
        emb = self.embedding(x)*torch.sqrt(self.dim)
        emb += self.pos_enc
        return emb #N, L, D
    
    def get_pos_encoding(self, dim, max_len):
        dim_loc = torch.arange(start=0, end=dim, step=2, device=info.device)
        pos_loc = torch.arange(start=0, end=max_len, step=1, device=info.device)
        pos_enc = torch.zeros((max_len, dim), device=info.device)
        
        denominator = torch.exp((-(dim_loc/dim))*(torch.log(torch.tensor(10000))))
        sin_pe = torch.sin(pos_loc.unsqueeze(1)*denominator.unsqueeze(0))
        cos_pe = torch.cos(pos_loc.unsqueeze(1)*denominator.unsqueeze(0))
        
        pos_enc[:,0::2] = sin_pe
        pos_enc[:,1::2] = cos_pe
        return pos_enc #L, D
    
    def initialization(self):
        nn.init.xavier_uniform_(self.embedding.weight)
    
class EncoderLayer(nn.Module):
    def __init__(self,) -> None:
        super(EncoderLayer, self).__init__()
        
    def forward(self, ):
        """"""
        
class DecoderLayer(nn.Module):
    def __init__(self, ) -> None:
        super(DecoderLayer, self).__init__()
        
    def forward(self, ):
        """"""