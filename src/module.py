from torch import nn
from layers import EmbeddingWithPosition, EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, N, vocab_size, pos_max_len, d_model, head, d_k, d_v, d_ff, drop_rate, shared_parameter, device):
        super(Encoder, self).__init__()
        self.emb_layer = EmbeddingWithPosition(vocab_size=vocab_size, pos_max_len=pos_max_len, embedding_dim=d_model, drop_rate=drop_rate, shared_parameter=shared_parameter, device=device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(head=head, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate) for n in range(N)])
    
    def forward(self, x, masked_info):
        """
        **INPUT SHAPE**
        x - N, L
        """
        output = self.emb_layer(x) #N, L, D
        for enc_layer in self.encoder_layers:
            output = enc_layer(x=output, masked_info=masked_info)
        return output
    
    def initialization(self):
        # self.emb_layer.initialization()
        for enc_layer in self.encoder_layers:
            enc_layer.initialization()
        
class Decoder(nn.Module):
    def __init__(self, N, vocab_size, pos_max_len, d_model, head, d_k, d_v, d_ff, drop_rate, shared_parameter, device):
        super(Decoder, self).__init__()
        self.emb_layer = EmbeddingWithPosition(vocab_size=vocab_size, pos_max_len=pos_max_len, embedding_dim=d_model, drop_rate=drop_rate, shared_parameter=shared_parameter, device=device)
        self.decoder_layers = nn.ModuleList([DecoderLayer(head=head, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, drop_rate=drop_rate) for n in range(N)])
    
    def forward(self, x, src_tgt_masked_info, tgt_masked_info, encoder_output):
        """
        **INPUT SHAPE**
        x - N, L
        """
        output = self.emb_layer(x) #N, L, D
        for dec_layer in self.decoder_layers:
            output = dec_layer(x=output, src_tgt_masked_info=src_tgt_masked_info, tgt_masked_info=tgt_masked_info, encoder_output=encoder_output)
        return output
    
    def initialization(self):
        # self.emb_layer.initialization()
        for dec_layer in self.decoder_layers:
            dec_layer.initialization()