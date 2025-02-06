import pandas as pd
import numpy as np

file='cast.csv'
df=pd.read_csv(file)
print(df)
#displaying the top 10roews of the data

print(df.head(10))

#displaying the last 10 rows of the data

print(df.tail(10))
#displaying the columns names
print(df.columns)

#displaying the number of rows and columns
print(df.shape)

#count thenumber of null or NAN values in the data
count=0
print(df.isnull().sum().sum())
# #percentage of missing values in the data
# total= df.shape[0]*df.shape[1]
# missing=    df.isnull().sum().sum()
# print('total number of missing values',missing)
# print('percentage of missing values',(missing/total)*100)

#displaying the datatype of each column
print(df.dtypes)

#displaying null values in rows
print(df.isnull().sum())

#displaying null values in columns
print(df.isnull().sum(axis=1))

#displaying unique values in the data
print(df.nunique())
