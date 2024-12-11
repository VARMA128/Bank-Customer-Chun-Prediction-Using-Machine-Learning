Module 2: Exploration data analysis of visualization and training a model by given attributes

#import library packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
data=pd.read_csv(r"C:\Users\user2\Desktop\churn_modeling.csv") 
data = data.dropna() 
data.columns

pd.crosstab(data.Gender,data.Exited) 
pd.crosstab(data.Balance,data.Exited) 
pd.crosstab(data.IsActiveMember,data.Exited)

# Plot histogram for Balance
plt.title("Balance of the Customers")
plt.xlabel("Balance")
plt.ylabel("No of Customers")
plt.hist(data.Balance)
plt.show()

# Plot histogram for CreditScore
plt.title("CreditScore of the Customers")
plt.xlabel("CreditScore")
plt.ylabel("No of Customers")
plt.hist(data.CreditScore)
plt.show()

# Pie chart for a given variable
def PropByVar(df, variable):
    dataframe_pie = df[variable].value_counts()
    ax = dataframe_pie.plot.pie(figsize=(6,6), autopct='%1.2f%%', fontsize=12)
    ax.set_title(variable + ' \n', fontsize=15)
    return np.round(dataframe_pie/df.shape[0]*100, 2)

PropByVar(data, 'Geography')

# Box plot for Balance
fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(data.Balance, ax=ax)
plt.title("Balance")
plt.show()

# Pairplot for the dataframe
sns.pairplot(data)
plt.show()

# Heatmap for the correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(data.corr(), ax=ax, annot=True)
plt.show()

from sklearn.model_selection import train_test_split

# Define features and response variable
X = data.drop(labels='Exited', axis=1)
y = data.loc[:, 'Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))
