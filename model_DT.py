import pandas as pd
import numpy as np




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from imblearn.combine import SMOTETomek
from functions import preprocessing
import joblib


df=pd.read_csv('BankChurners.csv')

X_train_res,X_test, y_train_res, y_test=preprocessing(df)

model=DecisionTreeClassifier()

    
model.fit(X_train_res, y_train_res)
 

prediction = model.predict(X_test)

print('DecisionTreeClassifier train set score' + ": {:.4f}%".format(model.score(X_train_res, y_train_res) * 100))
print('DecisionTreeClassifier test set score' + ": {:.4f}%".format(model.score(X_test, y_test) * 100))

print(classification_report(y_test,prediction))

print(f'confusion_matrix : {confusion_matrix(y_test,prediction)}')

plot_confusion_matrix(model,X_test,y_test)


joblib.dump(model, 'model_DT.pkl')
DT = joblib.load('model_DT.pkl')
   
    
    
    