import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import classification_report


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)
from imblearn.combine import SMOTETomek

#Importing the dataset


header=st.container()
dataset=st.container()
EDA_DV=st.container()
corr=st.container()
model=st.container()




st.sidebar.markdown("### credit card Churn Prediction App")
st.sidebar.image('images.png')

@st.cache
def get_data(file_name):
    df = pd.read_csv(file_name)
    return df



with header:
    st.title("credit card churn prediction")

with dataset:
    st.markdown("* **Bankchurners dataset**")
    df = get_data('BankChurners.csv')
    dataset = st.selectbox("please select to explore dataset", ("head","shape", "nan_values", "describe","columns_name"))
    def dataset_info(dataset):
        if dataset=="head":
            t=st.write('df.head(5):')
            r=st.write(df.head(5))
        if dataset=="shape":
            t = st.write('shape of dataset :')
            r=st.write(df.shape)
        if dataset=="describe":
            t = st.write('df.describe():')
            r=st.write(df.describe())
        if dataset=="nan_values":
            t=st.write('df.isnull().sum()')
            r=st.write(df.isnull().sum())
        if dataset == "columns_name":
            t = st.write('df.columns:')
            r = st.write(df.columns)
        return t,r
    t,r=dataset_info(dataset)

with EDA_DV:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("* **Exploratory Data Analysis and Data Visualization**")
    df = df.drop([df.columns[21], df.columns[22]], axis=1)
    df = df.drop('CLIENTNUM', axis=1)
    #st.markdown("* **target column**")

    feature=st.selectbox("please select a feature to have some data visualization", ("Attrition_Flag","Customer_Age", "Gender","Education_Level","Marital_Status","Income_Category","Card_Category"))

    if feature=="Attrition_Flag":

        st.text("Attrition_Flag unique value:")
        st.write(list(df['Attrition_Flag'].unique()))
        col1,col2=st.columns(2)

        col1.subheader("Attrition Flag count plot")
        Attrition_Flag=df['Attrition_Flag']


        fig = sns.countplot(Attrition_Flag)
        col1.pyplot()
        col2.subheader('Proportion of Attrition_Flag')
        Attrition_Flag1 = df['Attrition_Flag'].value_counts()
        fig1=plt.pie(Attrition_Flag1, labels=Attrition_Flag1.index, autopct='%1.1f%%')
        col2.pyplot()
    if feature=="Customer_Age":
        col3, col4 = st.columns(2)
        col3.subheader("Customer Age Distribution")
        fig3=sns.distplot(df['Customer_Age'])
        col3.pyplot()

        col4.subheader('Boxplot')
        fig3=sns.boxplot(y=df['Customer_Age'])
        col4.pyplot()
    if feature=="Gender":

        st.text("Gender unique value:")
        st.write(df['Gender'].unique().tolist())
        Gender = st.selectbox("selectBox for Gender visualization:",("countplot", "pie"))
        if Gender=="countplot":

            col1,col2=st.columns(2)

            col1.subheader("Gender")
            fig = sns.countplot(df['Gender'])
            col1.pyplot()
            col2.subheader('Attrition_Flag by Gender')
            fig1=sns.countplot(x=df['Attrition_Flag'], hue=df['Gender'])
            col2.pyplot()
        if Gender=="pie":
            col1, col2 = st.columns(2)
            col1.subheader("Attrited Customer vs Gender")

            attrited_gender = df.loc[df["Attrition_Flag"] == "Attrited Customer", ["Gender"]].value_counts()
            fig=plt.pie(attrited_gender, labels=df['Gender'].value_counts().index, autopct='%1.1f%%',startangle=90)
            col1.pyplot()
            col2.subheader('Existing Customer vs Gender')
            existing_gender = df.loc[df["Attrition_Flag"] == "Existing Customer", ["Gender"]].value_counts()
            fig1=plt.pie(existing_gender, labels=df['Gender'].value_counts().index, autopct='%1.1f%%',startangle=90)
            col2.pyplot()
    if feature == "Education_Level":
        col1,col2=st.columns(2)
        col1.subheader('Proportion of Education Levels')
        Education_Level = df['Education_Level'].value_counts()
        fig=plt.pie(Education_Level, labels=Education_Level.index, autopct='%1.1f%%')
        col1.pyplot()
        col2.subheader('Existing and Attrited Customers by Gender')
        fig1=sns.countplot(x=df['Education_Level'], hue=df['Attrition_Flag'])
        col2.pyplot()
    if feature == "Marital_Status":
        col1,col2=st.columns(2)
        col1.subheader('Proportion of Marital_Status')
        Marital_Status = df['Marital_Status'].value_counts()
        fig=plt.pie(Marital_Status, labels=Marital_Status.index ,autopct='%1.1f%%')
        col1.pyplot()
        col2.subheader('Existing and Attrited Customers by Marital_Status')
        fig1=sns.countplot(x=df['Marital_Status'], hue=df['Attrition_Flag'])
        col2.pyplot()
    if feature == "Income_Category":
        col1,col2=st.columns(2)
        col1.subheader('Proportion of Income_Category')
        Income_Category = df['Income_Category'].value_counts()
        fig=plt.pie(Income_Category, labels=Income_Category.index, autopct='%1.1f%%')
        col1.pyplot()
        col2.subheader('Existing and Attrited Customers by Income_Category')
        fig1=sns.countplot(x=df['Income_Category'], hue=df['Attrition_Flag'])
        col2.pyplot()
    if feature == "Card_Category":
        col1, col2 = st.columns(2)
        col1.subheader('Card_Category')
        fig = sns.countplot(data=df, x='Card_Category')
        col1.pyplot()
        col2.subheader('Attrition_Flag by Card_Category')
        fig1 = sns.countplot(x=df['Card_Category'], hue=df['Attrition_Flag'])
        col2.pyplot()


with corr:
    st.markdown("* **Correlation using heatmap**")
    df_n=df.select_dtypes(exclude=[object])
    fig, ax = plt.subplots(figsize=(12, 12))
    fig1 = sns.heatmap(df_n.corr(), cmap=sns.color_palette("vlag"), annot=True)

    st.pyplot()




with model:
    st.markdown("* **Classification**")
    def preprocessing(df):
        df = df.copy()
        Education_Level = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']
        Income_Category = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
        # print(f'shape of dataset : {df.shape}')
        # drop unnecessary columns
        df = df.drop([df.columns[21], df.columns[22]], axis=1)
        df = df.drop('CLIENTNUM', axis=1)

        df = df.replace('Unknown', np.NaN)
        # fill nan values of ordinal columns with their mode
        df['Education_Level'] = df['Education_Level'].fillna('Graduate')
        df['Income_Category'] = df['Income_Category'].fillna('Less than $40K')

        df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
        df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'F' else 0)
        df['Education_Level'] = df['Education_Level'].apply(lambda x: Education_Level.index(x))
        df['Income_Category'] = df['Income_Category'].apply(lambda x: Income_Category.index(x))

        Card_Category = pd.get_dummies(df['Card_Category'], drop_first=False)
        Marital_Status = pd.get_dummies(df['Marital_Status'], drop_first=False)

        df = df.drop(['Marital_Status', 'Card_Category'], axis=1)
        df = pd.concat([df, Marital_Status, Card_Category], axis=1)
        X = df.drop('Attrition_Flag', axis=1)
        y = df['Attrition_Flag']

        # split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # scale the features

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # oversampling

        smote = SMOTETomek(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        return X_train_res, X_test, y_train_res, y_test

    df=get_data('BankChurners.csv')


    X_train_res, X_test, y_train_res, y_test = preprocessing(df)




    classifier= st.selectbox('Please select an classifier:',('RandomForest','DecisionTree','GradientBoosting'))

    if classifier== 'RandomForest':
        model = RandomForestClassifier()
        model.fit(X_train_res, y_train_res)
        prediction = model.predict(X_test)
        confusion_matrix=confusion_matrix(y_test, prediction)

        col1,col2=st.columns(2)

        col1.write('RandomForestClassifier train set score' + ": {:.4f}%".format(model.score(X_train_res, y_train_res) * 100))
        col1.write('RandomForestClassifier test set score' + ": {:.4f}%".format(model.score(X_test, y_test) * 100))
        report=classification_report(y_test, prediction)
        col1.write('Classification report:\n ')
        col1.text('>\n ' + report)
        col1.write('\n ')
        col1.write('\n ')
        col2.write('Confustion matrix:\n ')
        fig = plt.figure()
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
        #plt.ylabel("True Label")
        #plt.xlabel("Predicted Label")
        col2.pyplot(fig)

    elif classifier == 'DecisionTree':
        model = DecisionTreeClassifier()

        model.fit(X_train_res, y_train_res)

        prediction = model.predict(X_test)
        confusion_matrix = confusion_matrix(y_test, prediction)
        col1, col2 = st.columns(2)
        col1.write('DecisionTreeClassifier train set score' + ": {:.4f}%".format(
        model.score(X_train_res, y_train_res) * 100))
        col1.write('DecisionTreeClassifier test set score' + ": {:.4f}%".format(model.score(X_test, y_test) * 100))
        report = classification_report(y_test, prediction)
        col1.write('Classification report:\n ')
        col1.text('>\n ' + report)
        col1.write('\n ')
        col1.write('\n ')
        col2.write('Confustion matrix:\n ')
        fig = plt.figure()
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
        # plt.ylabel("True Label")
        # plt.xlabel("Predicted Label")
        col2.pyplot(fig)

    elif classifier == 'GradientBoosting':
        model = GradientBoostingClassifier()

        model.fit(X_train_res, y_train_res)

        prediction = model.predict(X_test)
        confusion_matrix = confusion_matrix(y_test, prediction)
        col1, col2 = st.columns(2)

        col1.write('GradientBoostingClassifier train set score' + ": {:.4f}%".format(
            model.score(X_train_res, y_train_res) * 100))
        col1.write('GradientBoostingClassifier test set score' + ": {:.4f}%".format(model.score(X_test, y_test) * 100))
        report = classification_report(y_test, prediction)
        col1.write('Classification report:\n ')
        col1.text('>\n ' + report)
        col1.write('\n ')
        col1.write('\n ')
        col2.write('Confustion matrix:\n ')
        fig = plt.figure()
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
        # plt.ylabel("True Label")
        # plt.xlabel("Predicted Label")
        col2.pyplot(fig)



