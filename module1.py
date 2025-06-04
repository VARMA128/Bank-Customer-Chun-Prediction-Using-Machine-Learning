Module 1: Data validation and pre-processing technique
#import library packages
 
import pandas as pd 
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

#Load given dataset
data=pd.read_csv(r"C:\Users\user2\Desktop\churn_modeling.csv")

#Before drop the given dataset:
#dataset shows first 5 rows
data.head()

#dataset shows last 5 rows
data.tail()

#shape means no.of rows and columns
data.shape()

#shows columns
data.columns

#Checking data type and information about 
data.info()
data['Age'].unique() 
data.IsActiveMember.unique()
data.Gender.unique() 
data.Geography.unique()
data.Surname.unique() 
data.HasCrCard.unique()
data.NumOfProducts.unique() 
data.Exited.unique()

#preprocessing 
#eleminating irrelavant columns(parameters)
data=data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)
data.head()

#encoding object type to int type
from sklearn.preprocessing import LabelEncoder
var_mod = ['Geography', 'Gender']
# Initialize LabelEncoder
le = LabelEncoder()
# Encode categorical variables
for col in var_mod:
    data[col] = le.fit_transform(data[col])
data.head()
#pre processing done


