"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from models.embedding.positional_encoding import PostionalEncoding
from models.layers.classification_head import ClassificationHead

from models.model.encoder import Encoder
  

class Transformer(nn.Module):

    def __init__(self, d_model, n_head, max_len, seq_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__() 
        self.device = device

        self.encoder_input_layer = nn.Linear(   
            in_features=1, 
            out_features=d_model 
            )
   
        self.pos_emb = PostionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) 
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head, 
                               ffn_hidden=ffn_hidden, 
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,n_classes=5)

    def forward(self, src ): 
        print('before layer: '+ str(src.size()) )
        src= self.encoder_input_layer(src)
        print('after layer: '+ str(src.size()) )
        src= self.pos_emb(src)
        print('after pos_emb: '+ str(src.size()) )
        enc_src = self.encoder(src) 
        cls_res = self.classHead(enc_src)
        print('after cls_res: '+ str(cls_res.size()) )
        return cls_res
