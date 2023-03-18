 
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details
    def forward(self, q, k, v ,e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        
        if self.details: print('in Scale Dot Product, k_t size: '+ str(k_t.size()))
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product


        if self.details: print('in Scale Dot Product, score size: '+ str(score.size()))
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        if self.details: print('in Scale Dot Product, score size after softmax : '+ str(score.size()))

        if self.details: print('in Scale Dot Product, v size: '+ str(v.size()))
        # 4. multiply with Value
        v = score @ v

        if self.details: print('in Scale Dot Product, v size after matmul: '+ str(v.size()))
        return v, score
