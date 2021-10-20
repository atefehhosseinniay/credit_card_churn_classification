import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



from sklearn.preprocessing import LabelEncoder







from imblearn.combine import SMOTETomek


def preprocessing(df):
    df=df.copy()
    Education_Level = ['Uneducated','High School','College' ,'Graduate', 'Post-Graduate','Doctorate']
    Income_Category = ['Less than $40K', '$40K - $60K','$60K - $80K' , '$80K - $120K','$120K +']
    #print(f'shape of dataset : {df.shape}')
    # drop unnecessary columns
    df=df.drop([df.columns[21],df.columns[22]],axis=1)
    df=df.drop('CLIENTNUM',axis=1)
    
    df = df.replace('Unknown', np.NaN)
    # fill nan values of ordinal columns with their mode
    df['Education_Level'] = df['Education_Level'].fillna('Graduate')
    df['Income_Category'] = df['Income_Category'].fillna('Less than $40K')
    
    
    df['Attrition_Flag']=df['Attrition_Flag'].apply(lambda x: 1 if x=='Attrited Customer' else 0)
    df['Gender']=df['Gender'].apply(lambda x: 1 if x=='F' else 0)
    df['Education_Level'] = df['Education_Level'].apply(lambda x: Education_Level.index(x))
    df['Income_Category'] = df['Income_Category'].apply(lambda x: Income_Category.index(x))
    
    Card_Category = pd.get_dummies(df['Card_Category'], drop_first=False)
    Marital_Status = pd.get_dummies(df['Marital_Status'], drop_first=False)
    
    df = df.drop(['Marital_Status','Card_Category'], axis=1)
    df = pd.concat([df,Marital_Status,Card_Category],axis=1)
    X=df.drop('Attrition_Flag',axis=1)
    y=df['Attrition_Flag']
    
    #split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #scale the features
   
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # oversampling

    smote=SMOTETomek(random_state=42)
    X_train_res, y_train_res=smote.fit_resample(X_train,y_train)
    
    return X_train_res,X_test, y_train_res, y_test



    
    


    

    


    
