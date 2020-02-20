from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import torch.nn.functional as F
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim


def getmodels():
    '''
        getmodels() 
        in:     None
        return: None
        Downloads models trained on google colab models in google drive 
    '''
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
    
    drive = GoogleDrive(gauth)
    
    file_list = drive.ListFile({'q': "'1Xzt8iuAKoQxE1Tx5pylft4oPknmjseQJ' in parents and trashed=false"}).GetList()
    
    for file1 in file_list:
        if "model" in file1['title']:
            print('Downloading title: %s, id: %s' % (file1['title'], file1['id']))
            file1.GetContentFile(file1['title'])

# dataloader
class Football(Dataset):
    '''
        Football() - Pytorch Dataloader  
        in: Football dataset from hackerearch 
        return: Catagorical data cols ; Numerical Data Columns; Binary Target    
        Downloads models trained on google colab models in google drive 
    '''
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:, embedded_col_names].copy(
        ).values.astype(np.int64)  # categorical columns
        self.X2 = X.drop(
            columns=embedded_col_names).copy().values.astype(
            np.float32)  # numerical columns
        self.y = Y.to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def process_training(train_data):
    '''
        NOT USED HERE
        process_training() - Process training data 
        in: traning data path [str] 
        return: Catagorical data cols ; Numerical Data Columns; Binary Target    
        Processes training data   
    '''
    football = pd.read_csv(dir + train_data)
    football_x = football.drop("Won_Championship", axis=1)
    football_x = football_x.drop("ID", axis=1)

    football_y = football["Won_Championship"].copy()

    num_attributes = football_x.select_dtypes(exclude='object')
    cat_attributes = football_x.select_dtypes(include='object')

    num_attributes = list(num_attributes)
    cat_attributes = list(cat_attributes)

    num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attributes), ])

    for col in cat_attributes:
        football_x[col] = LabelEncoder().fit_transform(football_x[col])
        football_x[col] = football_x[col].astype('category')

    embedded_cols = {
        n: len(
            col.cat.categories) for n,
        col in football_x[cat_attributes].items() if len(
            col.cat.categories) > 2}
    emb_cols = embedded_cols.keys()  # names of columns chosen for embedding
    # embedding sizes for the chosen columns
    emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in embedded_cols.items()]
    embedded_col_names = embedded_cols.keys()

    return None


def process_test(test_data):
    '''
        NOT USED HERE
        process_training() - Process training data 
        in: traning data path [str] 
        return: Catagorical data cols ; Numerical Data Columns; Binary Target    
        Processes training data   
    '''
    football = pd.read_csv(test_data)
    football_ID = football["ID"].copy()
    football_x = football.drop("ID", axis=1)

    num_attributes = football_x.select_dtypes(exclude='object')
    cat_attributes = football_x.select_dtypes(include='object')

    num_attributes = list(num_attributes)
    cat_attributes = list(cat_attributes)

    num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])
    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attributes), ])

    for col in cat_attributes:
        football_x[col] = LabelEncoder().fit_transform(football_x[col])
        football_x[col] = football_x[col].astype('category')

    embedded_cols = {
        n: len(
            col.cat.categories) for n,
        col in football_x[cat_attributes].items() if len(
            col.cat.categories) > 2}
    emb_cols = embedded_cols.keys()  # names of columns chosen for embedding
    # embedding sizes for the chosen columns
    emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in embedded_cols.items()]
    embedded_col_names = embedded_cols.keys()

    batch_size = football_x.shape[0]
    test_ds = Football(football_x, football_ID, embedded_col_names)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return football_x, football_ID, test_dl


def nn_probs(model, test_dl, device):
    '''
        nn_probs() - Given features, returns a softmax probability distribution 
        in: traning data path [str] 
        return: Catagorical data cols ; Numerical Data Columns; Binary Target    
        Processes training data   
    '''
    preds = []
    model.eval()
    with torch.no_grad():
        for x_cat, x_num, _ in test_dl:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            out = model(x_cat, x_num)
            #probability of a positive classification
            prob = F.softmax(out, dim=1)[:,1]
            #_, pred  = torch.max(prob, 1)
            preds.extend(prob.detach().numpy())
    return preds

def process_test_ml_models(test_data):
    '''
        process_test_ml_models() - Given features, returns a softmax probability distribution 
        in: test data path [str] 
        return: Catagorical data cols ; Numerical Data Columns; Binary Target    
        Processes training data   
    '''

    test_dir = test_data
    football = pd.read_csv(test_dir)
    football_test_x  = football.drop('ID',axis=1)
    num_attributes = football_test_x.select_dtypes(exclude='object')
    cat_attributes = football_test_x.select_dtypes(include='object')
    num_attributes = list(num_attributes)
    cat_attributes = list(cat_attributes)
    
    num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
    
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attributes),
        ("cat",OneHotEncoder(),cat_attributes)
    ])
    
    football_x_prepared = full_pipeline.fit_transform(football_test_x)

    return football_x_prepared 





