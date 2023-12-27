import pickle
import pandas as pd 
import numpy as np

with open('DebtPredictor/model_debt.pkl', 'rb') as f:
    model = pickle.load(f)

new_data = {
    'Loan_Amount': [10000],
    'Shopping_Expenses': [800],
    'Travel_Expenses': [200],
    'Movie_Expenses': [10],
    'Dining_Expenses': [250],
    'Entertainment_Expenses': [150],
    'Healthcare_Expenses': [200],
}

new_df = pd.DataFrame(new_data)

# Make predictions using the loaded model
predictions = model.predict(new_df)
probabilty = np.max(model.predict_proba(new_df))
print(predictions)
print(probabilty)
