import numpy as np
class MyLogisticRegression:
    def __init__(self,learn_rate=0.001,epochs=1000):
        self.learn_rate=learn_rate
        self.epochs=epochs
           

    def sigmoid(x,weights,bias):
        return 1/(1+ np.exp(-(np.dot(x,weights)+bias)))


    def cost_fn(self,h,y,x):
        m=x.shape[0]
        cost=(-1/m)*(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
        return cost
    def gradient_fn(self,weights,bias,x,y):
        m=x.shape[0]
        for i in range(self.epochs):
            h=sigmoid(x,weights,bias)
            cost=self.cost_fn(self,h,y,x)

            dw=(1/m) * np.dot(x.T,(h-y))
            db=(1/m) * np.sum(h-y)
            w=w- self.learn_rate * dw
            b= b- self.learn_rate *db
            return w,b

