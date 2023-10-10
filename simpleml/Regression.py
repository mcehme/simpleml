import numpy as np

class LinearRegression:
    def __h(self, x):
        return x@self.W
    def fit(self, X, y, epochs=1000, alpha=0.05):
        if(y.ndim == 1):
            y = np.expand_dims(y, 1)
        m = np.size(X, 0)
        n = np.size(X, 1)
        self.W = np.zeros((n+1, 1))
        X = np.hstack((np.ones((m, 1)), X))
        X_T = np.transpose(X)

        gradient = lambda y, x: 1/m*X_T@(self.__h(x)  - y)

        for _ in range(epochs):
            self.W = self.W - alpha*gradient(y, X)
        return
        
    def predict(self, X):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return self.__h(X)

class LogisticRegression():
    def __h(self, x):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        return sigmoid(x@self.W)
    def fit(self, X, y, alpha=0.05, epochs=1000):
        # prepping a few variables
        if(y.ndim == 1):
            y = np.expand_dims(y, 1)
        m = np.size(X, 0)
        n = np.size(X, 1)
        self.W = np.zeros((n+1, 1))
        X = np.hstack((np.ones((m, 1)), X))
        X_T = np.transpose(X)

        gradient = lambda y, x: 1/m*X_T@(self.__h(x) - y)
        

        for _ in range(epochs):
            self.W = self.W - alpha*gradient(y, X)
        return
    
    def predict(self, X):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return np.round(self.__h(X))