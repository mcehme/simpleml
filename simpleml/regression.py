import numpy as np

class LinearRegression:
    '''A Linear Regression class using gradient descent'''
    def __h(self, x):
        '''Calculates the predicted values using the current weights
        
        :param x: the input data
        :rtype: ndarray
        :return: A numpy ndarray representing the predicted values'''
        return x@self.W
    

    def fit(self, X, y,  alpha=0.05, epochs=100):
        '''Fits the linear regression to the given training data
        
        :param X: The input data 
        :param y: The target data
        :param alpha: The learning rate (defaults to 0.05)
        :param epochs: Number of epochs to train for (defaults to 100)
        :rtype: NoneType
        :return: None'''


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
        '''Takes input data and computes predicted output
        
        :param X: the input data
        :rtype: ndarray
        :return: A numpy ndarry of the predicted values'''
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return self.__h(X)

class LogisticRegression():
    def __h(self, x):
        '''Calculates the predicted probabilities using the current weights
        
        :param x: the input data
        :rtype: ndarray
        :return: A numpy ndarray representing the predicted values'''
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        return sigmoid(x@self.W)
    def fit(self, X, y, alpha=0.05, epochs=1000):
        '''Fits the logistic regression to the given training data
        
        :param X: The input data 
        :param y: The target data
        :param alpha: The learning rate (defaults to 0.05)
        :param epochs: Number of epochs to train for (defaults to 1000)
        :rtype: NoneType
        :return: None'''
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
        '''Takes input data and computes predicted classes
        
        :param X: the input data
        :rtype: ndarray
        :return: A numpy ndarry of the predicted classes'''
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        return np.round(self.__h(X))