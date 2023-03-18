import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

 
from sklearn.model_selection import train_test_split
 
class EcgDataset(Dataset):
    def __init__(self,x,y)  :
        super().__init__()
        #file_out = pd.read_csv(fileName)
        #x = file_out.iloc[:,:-1].values
        #y = file_out.iloc[:,-1:].values 
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index): 
        return self.X[index].unsqueeze(1), self.Y[index] 
    

class MyTestDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size 
        file_out_test = pd.read_csv('mitbih_test.csv')
        
        x_test = file_out_test.iloc[:,:-1].values
        y_test = file_out_test.iloc[:,-1:].astype(dtype=int).values   
 
        test_set= EcgDataset(x= x_test, y= y_test) 
        self.dataLoader= DataLoader(test_set, batch_size=self.batch_size, shuffle=True,  ) 

    def getDataLoader(self): 
        return self.dataLoader

class myDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        file_out_train = pd.read_csv('mitbih_train.csv') 

        x_train = file_out_train.iloc[:,:-1].values
        y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values 
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15 )  

        #print("x_train shape on batch size =  " + str(x_train.shape ))
        #print('x_val shape on batch size =  ' + str(x_val.shape))
        #print('y_train shape on batch size =  '+ str(y_train.shape ))
        #print('y_val shape on batch size =  ' + str( y_val.shape) )

        train_set= EcgDataset(x= x_train, y= y_train) 

        val_set= EcgDataset(x= x_val, y= y_val) 

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )
        }
        self.dataloaders = dataloaders
        

    def getDataLoader(self): 
        return self.dataloaders
 
# using loadtxt()
#test_set = pd.read_csv("mitbih_test.csv")

#train_test = pd.read_csv("mitbih_train.csv" )
#label_names = ['N','S','F','V','Q']

#x_test = test_set.iloc[:,:-1].values
#y_test = test_set.iloc[:,-1:].astype(dtype=int).astype(dtype=str).values

#print((y_test[-10:,:]))
#y_test_one = np.zeros(shape= (y_test.shape[0], len(label_names)), dtype=float)
#for i in range(len(label_names)):  
#    y_test[y_test == str(i)] = label_names[i]
    #indexes = np.where((y_test  == i).all(axis=1))  
    #y_test_one[indexes,i]  = 1
#y_test = y_test.reshape(y_test.shape[0], 1 ) 
#print(y_test.shape)
#print(x_test.shape)
#test_data = {'ecg': x_test, 'labels': y_test}
#data = pd.DataFrame(test_data) 
#data.head() 