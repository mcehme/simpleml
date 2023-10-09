import numpy as np

class LinearRegression:
    def __h(self, x):
        return x@self.W
    def fit(self, X, y, epochs, alpha) -> None:
        y = np.expand_dims(y, 1)
        m = np.size(X, 0)
        n = np.size(X, 1)
        self.W = np.zeros((n+1, 1))
        X = np.hstack((np.ones((m, 1)), X))
        X_T = np.transpose(X)

        gradient = lambda x, y: 1/m*X_T@(self.__h(x)  - y)

        for _ in range(epochs):
            self.W = self.W - alpha*gradient(y, X)
        return
        
    def predict(self, X):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return np.squeeze(self.__h(X))

class LogisticRegression():
    def __h(self, x):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        return sigmoid(x@self.W)
    def fit(self, X, y, alpha=0.05, epochs=1000):
        # prepping a few variables
        y = np.expand_dims(y, 1)
        m = np.size(X, 0)
        n = np.size(X, 1)
        self.W = np.zeros((n+1, 1))
        X = np.hstack((np.ones((m, 1)), X))
        X_T = np.transpose(X)

        gradient = lambda x, y: 1/m*X_T@(self.__h(x)  - y)
        

        for _ in range(epochs):
            self.W = self.W - alpha*gradient(y, X)
        return
    
    def predict(self, X):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return np.round(np.squeeze(self.__h(X)))