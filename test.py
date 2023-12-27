import pickle
import pandas as pd 
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

data = {'m1': [60], 'm2': [70]}
input_df = pd.DataFrame(data)

y_pred= model.predict(input_df)
probabilty = np.max(model.predict_proba(input_df))
print(y_pred) # prediction value ( 1- > pass , 0 -> fail)
print(probabilty) # probability score


