# TimeSeriesClassification
This repository is created for experiments on time series classification. 

Modifications made on the traditional Transformer model used for NLP. 
Traditional transformer models get a sequence as input and I take the advantage of this sequence input.


You can find hyper parameters in model_summary.py file. 
The output for model summary is represented below. The dimensions are batch

before input layer: torch.Size([7, 136, 1])
after input layer: torch.Size([7, 136, 100])
after pos_emb: torch.Size([7, 136, 100])
in Multi Head Attention Q,K,V: torch.Size([7, 136, 100])
in splitted Multi Head Attention Q,K,V: torch.Size([7, 4, 136, 25])
in Scale Dot Product, k_t size: torch.Size([7, 4, 25, 136])
in Scale Dot Product, score size: torch.Size([7, 4, 136, 136])
in Scale Dot Product, score size after softmax : torch.Size([7, 4, 136, 136])
in Scale Dot Product, v size: torch.Size([7, 4, 136, 25])
in Scale Dot Product, v size after matmul: torch.Size([7, 4, 136, 25])
in Multi Head Attention, score value size: torch.Size([7, 4, 136, 25])
in Multi Head Attention, score value size after concat : torch.Size([7, 136, 100])
in encoder layer : torch.Size([7, 136, 100])
in encoder after norm layer : torch.Size([7, 136, 100])
in encoder after ffn : torch.Size([7, 136, 100])
in Multi Head Attention Q,K,V: torch.Size([7, 136, 100])
in splitted Multi Head Attention Q,K,V: torch.Size([7, 4, 136, 25])
in Scale Dot Product, k_t size: torch.Size([7, 4, 25, 136])
in Scale Dot Product, score size: torch.Size([7, 4, 136, 136])
in Scale Dot Product, score size after softmax : torch.Size([7, 4, 136, 136])
in Scale Dot Product, v size: torch.Size([7, 4, 136, 25])
in Scale Dot Product, v size after matmul: torch.Size([7, 4, 136, 25])
in Multi Head Attention, score value size: torch.Size([7, 4, 136, 25])
in Multi Head Attention, score value size after concat : torch.Size([7, 136, 100])
in encoder layer : torch.Size([7, 136, 100])
in encoder after norm layer : torch.Size([7, 136, 100])
in encoder after ffn : torch.Size([7, 136, 100])
in classification head : torch.Size([7, 136, 100])
in classification head after seq: torch.Size([7, 5])
after cls_res: torch.Size([7, 5]) 
![image](https://user-images.githubusercontent.com/6734818/225657838-b3b211b1-9412-4752-ab98-059051f61060.png)


