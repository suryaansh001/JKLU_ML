# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
df=pd.read_csv('House_Rent_Dataset.csv')
print(df)

# %%
import random
random.seed(42)
random.shuffle(df.values)
print(df)

# %%
print(df.columns)
print(df.info())
#dropping irrelevant columns
df.drop(['Posted On','Point of Contact','Tenant Preferred'],axis=1,inplace=True)
print(df.columns)

# %%
df.dropna()
print(df.info())

# %%
print(df['Furnishing Status'])
df['Furnishing Status'].replace(['Unfurnished','Semi-Furnished','Furnished'],[0,1,2],inplace=True)
print(df['Furnishing Status'])

# %%
print(df['Area Type'].unique())
df['Area Type'].replace(['Super Area' ,'Carpet Area' ,'Built Area'],[0,1,2],inplace=True)
print(df['Area Type'].unique())

# %% [markdown]
# same thing for the  
#     Area Locality      4746 non-null   object
#    City               4746 non-null   object
#    Furnishing Status  4746 non-null   object

# %%
#  Area Locality      4746 non-null   object
#    City               4746 non-null   object


print(df['Area Locality'].unique())
print(df['City'].unique())
print(df['City'].value_counts().count())
print(df['Area Locality'].value_counts().count())
#replacing with integers
for i in range(df['City'].value_counts().count()):
    df['City'].replace(df['City'].unique()[i],i,inplace=True)
for i in range(df['Area Locality'].value_counts().count()):
    df['Area Locality'].replace(df['Area Locality'].unique()[i],i,inplace=True)
print(df['Area Locality'].unique())
print(df['City'].unique())


# %%
print(df.info())
print(df['Floor'].unique())
for i in range(df['Floor'].unique().size):
    df['Floor'].replace(df['Floor'].unique()[i],i,inplace=True)
print(df['Floor'].unique())


# %%
print(df.info())

# %%
print(df.isnull().sum())
print(df.info())

# %%
test=df.tail(1000)

# %%
print(df.shape)

# print(df.iloc[0:df.shape[0]-1000])
remaining=df.iloc[0:df.shape[0]-1000]
print("shape of remaining",remaining.shape)

# train=remaining.iloc[0:int(remaining.shape[0]*0.8)]
# # print("train",train)
# print("shape of train",train.shape)
# validation=remaining.iloc[int(remaining.shape[0]*0.8):remaining.shape[0]]
# # print("validation",validation)
# print("shape of validation",validation.shape)

# %%
#splitting using sklearn 80% of remaining is train and 20% is test data random seeed 42
from sklearn.model_selection import train_test_split
train,validation=train_test_split(remaining,test_size=0.2,random_state=42)
print("shape of train",train.shape)
print("shape of test",validation.shape)

# %%
print(train.head(5))


# %%
import random
random.seed(42)
random.shuffle(train.values)
print(train)

# %%
plt.scatter(train['Size'],train['Rent'])
plt.xlabel('Size')
plt.ylabel('Rent')
plt.title('Size vs Rent')
plt.show()
plt.savefig('Size vs Rent1.png')

# %%
# #plotting with relevant scale ans after normalisatino
# maxx=train['Size'].max()
# minn=train['Size'].min()
# train['Size']=(train['Size']-minn)/(maxx-minn)
# maxx=train['Rent'].max()
# minn=train['Rent'].min()
# train['Rent']=(train['Rent']-minn)/(maxx-minn)

# plt.scatter(train['Size'],train['Rent'])
# plt.xlabel('Size')
# plt.ylabel('Rent')
# plt.title('Size vs Rent')

# plt.show()
# plt.savefig('Size vs Rent.png')

# %%
# plt.bar(train['Size'],train['Rent'])
# plt.show()

# %%
#2. Find average rent prices in different cities and report which city has the highest average rent
print("average rent prices in different cities")
print(train.groupby('City')['Rent'].mean())
print("city with highest average rent")
print(train.groupby('City')['Rent'].mean().idxmax())


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = LinearRegression()
x=train['Size'].values.reshape(-1,1)
Y=train['Rent'].values.reshape(-1,1)
model=model.fit(x,Y)
y_pred = model.predict(x)
print("MAE:", mean_absolute_error(Y, y_pred))
print("RMSE:", mean_squared_error(Y, y_pred))

# %%
#plotting ypred=red and y=blue
plt.scatter(train['Size'],train['Rent'],color='blue')
plt.scatter(train['Size'],y_pred,color='red')
plt.xlabel('Size')
plt.ylabel('Rent')
plt.title('Size vs Rent')
plt.show()

# %%
print(y_pred)
print(Y)

# %%
def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

int(rmse(Y, y_pred))

# %%
#now in train2 taking only 2 columns size and floor
train2=train[['Size','BHK']]
model2 = LinearRegression()
x=train2.values
Y=train['Rent'].values.reshape(-1,1)
model2=model2.fit(x,Y)
y_pred = model2.predict(x)
print("MAE:", mean_absolute_error(Y, y_pred))
print("RMSE:", mean_squared_error(Y, y_pred))


# %%
#now in train2 taking only 2 columns size and floor
train2=train[['Size','Floor']]
model2 = LinearRegression()
x=train2.values
Y=train['Rent'].values.reshape(-1,1)
model2=model2.fit(x,Y)
y_pred = model2.predict(x)
print("MAE:", mean_absolute_error(Y, y_pred))
print("RMSE:", mean_squared_error(Y, y_pred))


# %%
print(train.columns)

# %%
lis=[]
for i in train.columns:
    for j in train.columns:
        if i!=j:
            model2 = LinearRegression()
            x=train[[i,j]].values
            Y=train['Rent'].values.reshape(-1,1)
            model2=model2.fit(x,Y)
            y_pred = model2.predict(x)
            mae=mean_absolute_error(Y, y_pred)
            Rmse=rmse(Y, y_pred)
            lis.append([mae,Rmse,i,j])
            # print("MAE:", mean_absolute_error(Y, y_pred))
            # print("RMSE:", mean_squared_error(Y, y_pred))
            # print(i,j)
# print(lis)
lis.sort()
lis=np.array(lis)
print("least error is ",float(lis[0][0])," ",lis[0][-2]," ",lis[0][-1])
print("least error is ",float(lis[1][0])," ",lis[1][-2]," ",lis[1][-1])
print("least error is ",float(lis[2][0])," ",lis[2][-2]," ",lis[2][-1])


# %%



