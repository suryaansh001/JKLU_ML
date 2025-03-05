# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random as rand

# %% [markdown]
# 1. Randomly shuffle the dataset by taking a random seed of “42”. Create a training and testing set
# partitions in the ratio of 70% : 30% by taking last 30% rows in the test set. The remaining rows will be
# the training set. Make sure that the columns have the same datatypes. Display the mean values for each
# columns and the number of samples belonging to each category (admitted and not-admitted)

# %%
rand.seed(42)
#giving the names to columns 
df=pd.read_csv('student_marks.csv')
# print(df.head(10))
df.columns=['Subject1','Subject2','Status']
# print(df.head(10))
print(df.describe())
print(df.info())
print(df.tail(10))

# %%

#converting the columns into int 
df['Subject1']=df['Subject1'].astype(int)
df['Subject2']=df['Subject2'].astype(int)
df['Status']=df['Status'].astype(int)

# %%
#mean of the columns
print(df['Subject1'].mean())
print(df['Subject2'].mean())
# print(df['Status'].mean())

# %%
print(df.shape)
#there are 99 rows 
#shuffling the datafra,e
np.random.seed(42)
np.random.shuffle(df.values)
# print(df.head(10))
test_set_size=int(0.3*df.shape[0])
train_set_size=df.shape[0]-test_set_size
print(test_set_size)
print(train_set_size)
train_Set=df.head(train_set_size)
test_Set=df.tail(test_set_size)
# print("train set",train_Set.head(10))
# print("test set",test_Set.head(10))
print("size of test set",test_Set.shape)
print("size of train set",train_Set.shape)



# %% [markdown]
# . Create a scatter plot using the training set and mark the points differently for different classes.

# %%
#scatter plot of test size

plt.scatter(test_Set['Subject1'],test_Set['Subject2'],c=test_Set['Status'])
plt.xlabel('Subject1')
plt.ylabel('Subject2')
plt.title('Test set')
plt.show()


# %%


# %%
from scipy.optimize import fmin_tnc

# %% [markdown]
# 

# %% [markdown]
# 3. Plot the decision boundary on the previously drawn scatter plot.

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fmin_tnc

# class MyLogisticRegression:
#     def __init__(self, learn_rate=0.001, epochs=1000):
#         self.learn_rate = learn_rate
#         self.epochs = epochs

#     def sigmoid(self, x, weights, bias):
#         return 1 / (1 + np.exp(-(np.dot(x, weights) + bias)))

#     def cost_fn(self, params, x, y):
#         weights = params[:-1]
#         bias = params[-1]
#         h = self.sigmoid(x, weights, bias)
#         m = x.shape[0]
#         cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
#         return cost

#     def gradient_fn(self, params, x, y):
#         weights = params[:-1]
#         bias = params[-1]
#         h = self.sigmoid(x, weights, bias)
#         m = x.shape[0]
#         dw = (1/m) * np.dot(x.T, (h - y))
#         db = (1/m) * np.sum(h - y)
#         grad = np.append(dw, db)
#         return grad

#     def fit(self, x, y):
#         m, n = x.shape
#         initial_params = np.zeros(n + 1)
#         x_with_bias = np.hstack([x, np.ones((m, 1))])
#         result = fmin_tnc(func=self.cost_fn, x0=initial_params, fprime=self.gradient_fn, args=(x_with_bias, y))
#         optimal_params = result[0]
#         self.weights = optimal_params[:-1]
#         self.bias = optimal_params[-1]

#     def predict(self, x):
#         m = x.shape[0]
#         x_with_bias = np.hstack([x, np.ones((m, 1))])
#         predictions = self.sigmoid(x_with_bias, self.weights, self.bias)
#         return (predictions >= 0.5).astype(int)


# %%


class MyLogisticRegression:
    def __init__(self,learn_rate=0.001,epochs=1000):
        self.learn_rate =learn_rate
        self.epochs =epochs

    def sigmoid(self,x,weights,bias):
        return 1 / (1 + np.exp(-(np.dot(x,weights) + bias)))

    def cost_fn(self,params,x,y):
        weights =params[:-1]
        bias =params[-1]
        h =self.sigmoid(x,weights,bias)
        m =x.shape[0]
        cost =(-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
        return cost

    def gradient_fn(self,params,x,y):
        weights =params[:-1]
        bias =params[-1]
        h =self.sigmoid(x,weights,bias)
        m =x.shape[0]
        dw =(1/m)*np.dot(x.T,(h-y))
        db =(1/m)*np.sum(h-y)
        grad =np.append(dw,db)
        return grad

    def fit(self,x,y):
        m,n =x.shape
        initial_params =np.zeros(n + 1)
        x_with_bias =np.hstack([x,np.ones((m,1))])
        result =fmin_tnc(func=self.cost_fn,x0=initial_params,fprime=self.gradient_fn,args=(x_with_bias,y))
        optimal_params =result[0]
        self.weights =optimal_params[:-1]
        self.bias =optimal_params[-1]

    def predict(self,x):
        m =x.shape[0]
        x_with_bias =np.hstack([x,np.ones((m,1))])
        predictions =self.sigmoid(x_with_bias,self.weights,self.bias)
        return (predictions >=0.5).astype(int)


# %%


# %% [markdown]
# 5. Create more columns in the dataframes (training and test) corresponding to higher order terms x 12,
# x22, and x1x2 .

# %%
#createin new columns
df['Subject1^2']=df['Subject1']**2
df['Subject2^2']=df['Subject2']**2
df['Subject1*Subject2']=df['Subject1']*df['Subject2']
print(df.head(10))
print(df.describe())
print(df.info())
print(df.tail(10))


