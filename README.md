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

==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
Transformer                                                  [7, 5]                    --
├─Linear: 1-1                                                [7, 136, 100]             200
├─PostionalEncoding: 1-2                                     [7, 136, 100]             --
│    └─Dropout: 2-1                                          [7, 136, 100]             --
├─Encoder: 1-3                                               [7, 136, 100]             --
│    └─ModuleList: 2-2                                       --                        --
│    │    └─EncoderLayer: 3-1                                [7, 136, 100]             143,812
│    │    └─EncoderLayer: 3-2                                [7, 136, 100]             143,812
├─ClassificationHead: 1-4                                    [7, 5]                    --
│    └─LayerNorm: 2-3                                        [7, 136, 100]             200
│    └─Sequential: 2-4                                       [7, 5]                    --
│    │    └─Flatten: 3-3                                     [7, 13600]                --
│    │    └─Linear: 3-4                                      [7, 512]                  6,963,712
│    │    └─ReLU: 3-5                                        [7, 512]                  --
│    │    └─Linear: 3-6                                      [7, 256]                  131,328
│    │    └─Linear: 3-7                                      [7, 5]                    1,285
==============================================================================================================
Total params: 7,384,349
Trainable params: 7,384,349
Non-trainable params: 0
Total mult-adds (M): 51.68
==============================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 20.03
Params size (MB): 29.54
Estimated Total Size (MB): 49.57
==============================================================================================================
