import pandas as pd
import numpy as np




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

from imblearn.combine import SMOTETomek
from function import preprocessing
import joblib


df=pd.read_csv('BankChurners.csv')

X_train_res,X_test, y_train_res, y_test=preprocessing(df)

model=RandomForestClassifier()

    
model.fit(X_train_res, y_train_res)
print('RandomForestClassifier' + ": {:.4f}%".format(model.score(X_test, y_test) * 100)) 

prediction = model.predict(X_test)

print('RandomForestClassifier train set score' + ": {:.4f}%".format(model.score(X_train_res, y_train_res) * 100))
print('RandomForestClassifier test set score' + ": {:.4f}%".format(model.score(X_test, y_test) * 100))

print(classification_report(y_test,prediction))

plot_confusion_matrix(model,X_test,y_test)

joblib.dump(model, 'model_RF.pkl')
RF = joblib.load('model_RF.pkl')
   
    
    
    