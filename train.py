import optuna , pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
df= pd.read_csv("marks.txt",sep=',')
df.columns =['m1','m2','result']
X=df.drop('result',axis=1)
y=df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
# print(X_test)

# def objective(trial):
#       n_estimators = trial.suggest_int('n_estimators', 2, 20)
#       max_depth = int(trial.suggest_int('max_depth', 1, 32))
#       clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
#       return model_selection.cross_val_score(clf, X_train, y_train, 
#            n_jobs=-1, cv=3).mean()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
# trial = study.best_trial
# # print('Accuracy: {}'.format(trial.value))
# # print("Best hyperparameters: {}".format(trial.params['n_estimators']))
# xgb =XGBClassifier(n_estimators= trial.params['n_estimators'] ,max_depth=trial.params['max_depth'])
# xgb.fit(X_train,y_train)
# with open('model.pkl','wb') as f:
#     pickle.dump(xgb,f)

# y_pred=xgb.predict(X_test)
# print('Accuracy: ', accuracy_score(y_pred,y_test))
