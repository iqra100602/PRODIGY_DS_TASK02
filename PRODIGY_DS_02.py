import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
df = pd.read_csv('train.csv')
df
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Pclass'] = df['Pclass'].astype('category')
df.drop_duplicates(inplace=True)
df.loc[df['Age'] < 0, 'Age'] = df['Age'].median()
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.head())
# Bar plot
sns.countplot(x='Survived', data=df)
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.title('Survival Count')
plt.show()
