import torch
from torch import nn
from module import Encoder, Decoder
import info

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, pos_max_len, d_model, head, d_k, d_v, d_ff, drop_rate):
        super(Transformer, self).__init__()
        self.share_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=info.PAD)
        nn.init.xavier_uniform_(self.share_embedding.weight)
        with torch.no_grad():
            self.share_embedding.weight[info.PAD].fill_(0)
        self.encoder = Encoder(N=N, vocab_size=vocab_size, pos_max_len=pos_max_len, d_model=d_model, head=head, 
                               d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate, shared_parameter=self.share_embedding)
        self.decoder = Decoder(N=N, vocab_size=vocab_size, pos_max_len=pos_max_len, d_model=d_model, head=head, 
                               d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate, shared_parameter=self.share_embedding)
        self.outputlayer = nn.Linear(d_model, vocab_size, bias=False)
        self.outputlayer.weight = self.share_embedding.weight
        self.initialization()
        
        self.vocab_size = vocab_size
        
    def forward(self, src_input, tgt_input):
        N_in_step, L_in_step = src_input.shape
        src_mask = self.get_mask(src_input)
        tgt_mask = self.get_mask(tgt_input)
        src_tgt_mask = (torch.bmm(tgt_mask.float(), src_mask.float()) == 0)
        encoder_output = self.encoder(x=src_input, masked_info=src_mask)
        decoder_output = self.decoder(x=tgt_input, src_tgt_masked_info=src_tgt_mask, tgt_masked_info=tgt_mask, encoder_output=encoder_output)
        predict = self.outputlayer(decoder_output).reshape(N_in_step*L_in_step, self.vocab_size)
        return predict
    
    def get_mask(self, x):
        mask = torch.zeros(x.shape).to(info.device)
        mask[x != 0] = 1.
        mask = (torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1)) == 0)
        return mask
    
    def initialization(self):
        self.encoder.initialization()
        self.decoder.initialization()