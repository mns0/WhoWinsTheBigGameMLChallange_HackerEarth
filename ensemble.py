import os.path
from os import path
from shutil import copyfile
import torch
import numpy as np
import glob
import tabular_model as Net2
import process_dataset as process
import pandas as pd
from joblib import load 
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier


###### GET models trained on Google Colab ######
GETMODELS = False
if GETMODELS:
    process.getmodels()

testds = 'test.csv'
trainds = 'train.csv'
PATH_DS = '../'

football_x, football_ID, test_dl = process.process_test(PATH_DS + testds)

#### Get the results from NN models ####
#### load NN Models ############
MODELS_PATH = glob.glob("*pth")
MODELS_PATH = ["model_3.pth"] 

# hard code embedding sizes
n_count = 4
embedding_sizes = [(3, 2), (4, 2), (10, 5), (3, 2)]
model = Net2.Net2_deep(embedding_sizes, n_count)
device = torch.device('cpu')

probs = np.zeros((len(football_ID), len(MODELS_PATH)))

print(MODELS_PATH)

for i, PATH in enumerate(MODELS_PATH):
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    tmp = process.nn_probs(model, test_dl, device)
    probs[:, i] = process.nn_probs(model, test_dl, device)


fig, ax  = plt.subplots()
#average probability
nn_consensus = np.mean(probs,axis=1)

######## Import XGBoost Model ############
xgboost = load("gbr_model_final.joblib")
ml_data_x = process.process_test_ml_models(PATH_DS+testds)
#xgboost_predicts = xgboost.predict_proba(ml_data_x)[:,1]
###

######## Import LGBOOST Model ############
lgboost = load("lbg_model_final.joblib")
lgboost_predicts = lgboost.predict_proba(ml_data_x)[:,1]
###

#avg models 
#models_avg = (nn_consensus + xgboost_predicts + lgboost) / 3
models_avg = (nn_consensus + lgboost_predicts) / 2
models_avg[models_avg >= 0.5] = 1
models_avg[models_avg < 0.5] = 0

### return results of the ensemble model
retdata = np.asarray([football_ID,models_avg]).T
ret_frame = pd.DataFrame(data=retdata,columns=['ID','Won_Championship'])

#rfname = 'single_r0e150_nnmodel_deepnet_submission_02182020.csv'
rfname = 'lightgbm_nnmodel_submission_02182020.csv'
if path.isfile(rfname): 
    copyfile(rfname, rfname + '.BAK')
    ret_frame.to_csv(rfname,index=False)
else:
    ret_frame.to_csv(rfname,index=False)


