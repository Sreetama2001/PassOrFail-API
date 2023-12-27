import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df= pd.read_csv("DebtPredictor\debt_data.csv")
df = df.drop(['encoded_0' ,  'encoded_1'], axis=1)
# print(df.info())
# print(df.describe())

X = df.drop('Loan_Default', axis=1)  # Features
y = df['Loan_Default']  # Target variable

# Correlation heatmap
'''
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)

with open('DebtPredictor/model_debt.pkl','wb+') as f:
    pickle.dump(rf_classifier,f)

y_pred = rf_classifier.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
