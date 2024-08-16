from torch import nn
from module import Encoder, Decoder
from layers import EmbeddingWithPosition


class Transformer(nn.Module):
    def __init__(self, vocab_size, dimension, ):
        super(Transformer, self).__init__()
        self.embedding = nn.EmbeddingWithPosition(num_embeddings=vocab_size, embedding_dim=dimension, padding_idx=0)
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, src_input, tgt_input):
        src_emb = self.embedding(src_input)
        encoder_outputs = self.encoder(src_emb)
        tgt_emb
        
        return