
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self,d_model, seq_len , details, n_classes: int = 5):
      super().__init__()
      self.norm = nn.LayerNorm(d_model)
      self.details = details
      #self.flatten = nn.Flatten()
      self.seq = nn.Sequential( nn.Flatten() , nn.Linear(d_model * seq_len , 512) ,nn.ReLU(),nn.Linear(512, 256)
                               ,nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_classes))
 
    def forward(self,x):

      if self.details:  print('in classification head : '+ str(x.size())) 
      x= self.norm(x)
      #x= self.flatten(x)
      x= self.seq(x)
      if self.details: print('in classification head after seq: '+ str(x.size())) 
      return x