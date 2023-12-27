import pickle
import pandas as pd 
import numpy as np

with open('DebtPredictor/model_debt.pkl', 'rb') as f:
    model = pickle.load(f)

new_data = {
    'Loan_Amount': [10000],
    'Shopping_Expenses': [8000],
    'Travel_Expenses': [200000],
    'Movie_Expenses': [10],
    'Dining_Expenses': [250],
    'Entertainment_Expenses': [150],
    'Healthcare_Expenses': [200],
}

new_df = pd.DataFrame(new_data)

# Make predictions using the loaded model
predictions = model.predict(new_df) # binary 
probabilty = np.max(model.predict_proba(new_df)) # 0 - 1 decimal
print(predictions) # class value 1 or 0 
print(probabilty) # class 
