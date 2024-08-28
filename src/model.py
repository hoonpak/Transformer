import torch
from torch import nn
from module import Encoder, Decoder
import info

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, pos_max_len, d_model, head, d_k, d_v, d_ff, drop_rate, device):
        super(Transformer, self).__init__()
        self.share_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.device = device
        nn.init.xavier_uniform_(self.share_embedding.weight)

        self.encoder = Encoder(N=N, vocab_size=vocab_size, pos_max_len=pos_max_len, d_model=d_model, head=head, 
                               d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate, shared_parameter=self.share_embedding, device=device)
        self.decoder = Decoder(N=N, vocab_size=vocab_size, pos_max_len=pos_max_len, d_model=d_model, head=head, 
                               d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate, shared_parameter=self.share_embedding, device=device)
        self.outputlayer = nn.Linear(d_model, vocab_size)
        self.outputlayer.weight = self.share_embedding.weight
        
        self.initialization()
        self.vocab_size = vocab_size
        
    def forward(self, src_input, tgt_input):
        pad_src_mask, attn_src_mask = self.get_mask(src_input)
        pad_tgt_mask, attn_tgt_mask = self.get_mask(tgt_input)
        src_tgt_mask = (torch.bmm(pad_tgt_mask.unsqueeze(2).float(), pad_src_mask.unsqueeze(1).float()) == 0) #N. TL, SL
        encoder_output = self.encoder(x=src_input, masked_info=attn_src_mask)
        decoder_output = self.decoder(x=tgt_input, src_tgt_masked_info=src_tgt_mask,
                                      tgt_masked_info=attn_tgt_mask, encoder_output=encoder_output)
        predict = self.outputlayer(decoder_output).reshape(-1, self.vocab_size)
        return predict
    
    def get_mask(self, x):
        mask = torch.zeros(x.shape).to(self.device)
        mask[x != 0] = 1. #N, L -> padding = 0, others = 1
        mask_output = (torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1)) == 0) #N, L, L
        return mask, mask_output
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()