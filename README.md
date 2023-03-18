# TimeSeries Classification and Forecasting
This repository was created for experiments on time series classification and forecasting.  
The underlying reason why transformer has been chosen for classification and forecastion of timeseries is transformers are faster than RNN-based models as all the input is ingested once. Training LSTMs is harder when compared with transformer networks, since the number of parameters is a lot more in LSTM networks.  
Currently, the model has been modified for the classification task. In the coming days, it will be further modified for forecasting.  
Transformer models used in NLP generally have an encoder and decoder for transforming input data (Query) into the expected output (Value). The main concept of transformer models is based on multi-head attention units, which generate a value representing the relation between Query and Key values within an energy matrix. The Query and Key values are generally identical to the input vector, and the multi-head attention mechanism generates a Value matrix that includes energy values representing the relation value between each point for the input vector (NxN).  

In this project, the Value matrice generated by multi head attention has been used for classification. A classification mechanism has been added to the end of the encoder part of the model. 
 

You can find hyper parameters in model_summary.py file. 
The output for model summary is represented below. 

![image](https://user-images.githubusercontent.com/6734818/225657838-b3b211b1-9412-4752-ab98-059051f61060.png)


TRANING.  
  
DATASET:   
https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=mitbih_train.csv. 

You can download easily MIT-BIH Arrhythmia Dataset from the link above. Download mitbih_train.csv and mitbih_test.csv files and paste in same path with 'ecg_dataset.py'              
There are 5 different classes and the all classes are enumerated from 0 to 4 in label dataset.  
batch_size =100   
seq_len= 187 # each ecg data sample contains 187 columns  
feature= 1 # we have only 1 feature beacuse of having 1D array
Input size for training : [batch_size, seq_len, feature]
TEST RESULTS: 
The batch size is assigned as 30 for testing. Also, you can see and compare target labels and predicted classes.  
![image](https://user-images.githubusercontent.com/6734818/226144528-31dea983-508c-4ee7-818f-c7a29607f242.png)       




