import numpy as np

class LinearRegression:
    def __init__(self):
        np.random.seed(1)
        # crating W matrix (2x1)
        self.W=np.random.random(2)

    # cost fuction to calculate error
    def costFunction(self,X,Y):
        n_sample=len(Y)
        Y_pred=self.W @ X
        return (1/(2*n_sample))*np.sum((Y_pred-Y)**2)
    
    def gardient_descent(self,X,Y,iterations=100,learning_rate=.01,n_sample=0):
        # creating an array of zeros for storing cost after every iteration
        costs = np.zeros((iterations,1))

        # equation of linear regression
        #                   y_pred=w0*x0+w1*x1          , where x0=1
        #                   y_pred=W @ X              ...(In matrix form)

        # printing initial cost
        print(f"Value of cost function before gardient descent: {self.costFunction(X,Y)}")

        # loop for applying gradient descent
        for i in range(iterations):
            # adjusting weights with every iteration
            self.W=self.W-(learning_rate/n_sample)*(X @ (self.W @ X -Y))
            costs[i]=self.costFunction(X,Y)
        
        # printing initial cost
        print(f"Value of cost function After {iterations} iterations of gardient descent: {self.costFunction(X,Y)}")

        return costs

    def fit(self,x,y,iterations=1000,learning_rate=.01):
        # finding sample size
        n_sample = len(y)

        # creating X matrix (1xn)
        X=np.array([np.ones((n_sample,)),np.array(x)])
        
        # creating Y matrix (1xn)
        Y=np.array(y)
        
        # crating W matrix (2x1)
        self.W=np.random.random(2)
        
        return self.gardient_descent(X,Y,iterations,learning_rate,n_sample)

    def predict(self,x):
        x=np.array(x)
        X=np.array([np.ones((len(x),)),x])
        return self.W @ X

    def params(self):
        return self.W
