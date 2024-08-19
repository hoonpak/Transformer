import torch
from torch import nn

import info
from sublayers import MultiHeadAttention, LayerLorm, PositionWiseFeedForward

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, pos_max_len, embedding_dim, drop_rate, shared_parameter):
        super(EmbeddingWithPosition, self).__init__()
        self.dim_sqrt = torch.sqrt(torch.tensor(embedding_dim))
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=info.PAD)
        self.dropout = nn.Dropout(p=drop_rate)
        
        self.embedding.weight = shared_parameter
        self.pos_enc = self.get_pos_encoding(dim=embedding_dim, max_len=pos_max_len) #L, D
        
    def forward(self, x):
        emb = self.embedding(x)*self.dim_sqrt
        emb += self.pos_enc
        emb = self.dropout(emb)
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
    
    # def initialization(self):
    #     nn.init.xavier_uniform_(self.embedding.weight)
    
class EncoderLayer(nn.Module):
    def __init__(self, head, d_model, d_k, d_v, d_ff, drop_rate):
        super(EncoderLayer, self).__init__()
        self.MHA_layer = MultiHeadAttention(head=head, d_model=d_model, d_k=d_k, d_v=d_v, is_masked=False)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.LN_layer1 = LayerLorm(d_model=d_model)
        
        self.PWFFN_layer = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.LN_layer2 = LayerLorm(d_model=d_model)

    def forward(self, x, masked_info):
        """
        **INPUT SHAPE**
        x - N, L, D
        masked_info - N, L, L -> padding = 0, else = 1
        [[False, False, False,  True,  True],
         [False, False, False,  True,  True],
         [False, False, False,  True,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True]]
        """
        MHA_output = self.drop1(self.MHA_layer(Q=x, K=x, V=x, masked_info=masked_info)) #N, L, d_m
        FF_input = self.LN_layer1(x+MHA_output) #N, L, d_m
        
        FF_output = self.drop2(self.PWFFN_layer(FF_input))
        encoder_output = self.LN_layer2(FF_input+FF_output) #N, L, d_m
        
        return encoder_output
    
    def initialization(self):
        self.MHA_layer.initialization()
        self.LN_layer1.initialization()
        self.PWFFN_layer.initialization()
        self.LN_layer2.initialization()
        
class DecoderLayer(nn.Module):
    def __init__(self, head, d_model, d_k, d_v, d_ff, drop_rate):
        super(DecoderLayer, self).__init__()
        self.Masked_MHA_layer = MultiHeadAttention(head=head, d_model=d_model, d_k=d_k, d_v=d_v, is_masked=True)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.LN_layer1 = LayerLorm(d_model=d_model)

        self.MHA_layer = MultiHeadAttention(head=head, d_model=d_model, d_k=d_k, d_v=d_v, is_masked=False)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.LN_layer2 = LayerLorm(d_model=d_model)
        
        self.PWFFN_layer = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.drop3 = nn.Dropout(p=drop_rate)
        self.LN_layer3 = LayerLorm(d_model=d_model)
        
    def forward(self, x, src_tgt_masked_info, tgt_masked_info, encoder_output):
        """
        **INPUT SHAPE**
        x - N, L, D
        src_tgt_masked_info - N, L, L -> padding = 0, else = 1
        [[False, False, False,  False,  True],
         [False, False, False,  False,  True],
         [False, False, False,  False,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True]]
        tgt_masked_info - N, L, L -> padding = 0, else = 1
        """
        Masked_MHA_output = self.drop1(self.Masked_MHA_layer(Q=x, K=x, V=x, masked_info=tgt_masked_info)) #N, L, d_m
        MHA_input = self.LN_layer1(x+Masked_MHA_output) #N, L, d_m
        
        MHA_output = self.drop2(self.MHA_layer(Q=MHA_input, K=encoder_output, V=encoder_output, masked_info=src_tgt_masked_info)) #N, L, d_m
        FF_input = self.LN_layer2(MHA_input+MHA_output) #N, L, d_m
        
        FF_output = self.drop3(self.PWFFN_layer(FF_input))
        decoder_output = self.LN_layer3(FF_input+FF_output) #N, L, d_m
        
        return decoder_output

    def initialization(self):
        self.Masked_MHA_layer.initialization()
        self.LN_layer1.initialization()
        self.MHA_layer.initialization()
        self.LN_layer2.initialization()
        self.PWFFN_layer.initialization()
        self.LN_layer3.initialization()