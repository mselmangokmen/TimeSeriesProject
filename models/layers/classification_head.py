
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self,d_model, seq_len , n_classes: int = 5):
      super().__init__()
      self.norm = nn.LayerNorm(d_model)

      #self.flatten = nn.Flatten()
      self.seq = nn.Sequential( nn.Flatten() , nn.Linear(d_model * seq_len , 512) ,nn.ReLU(),nn.Linear(512, 256),nn.Linear(256, n_classes))
 
    def forward(self,x):

      print('in classification head : '+ str(x.size())) 
      x= self.norm(x)
      #x= self.flatten(x)
      x= self.seq(x)
      print('in classification head after seq: '+ str(x.size())) 
      return x