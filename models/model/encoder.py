"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.positional_encoding import PostionalEncoding 


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details, device):
        super().__init__()

        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head
                                                  ,details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ): 
        for layer in self.layers:
            x = layer(x ) 
        return x
