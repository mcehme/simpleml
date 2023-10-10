import numpy as np
def mean_squared_error(y, y_pred):
    '''Calculate the mean squared error.
    
    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The mean square error'''
    return np.sum(np.square(y - y_pred))/np.size(y,0)
def r2(y, y_pred):
    '''Calculate the r2 value.
    Will calculate by column if there are multiple columns.
    
    :param y: the true values
    :param y_pred: the predicted values
    :rtype: ndarray
    :return: A numpy ndarray representing the r2 value for each column'''
    return np.squeeze(1 - np.sum(np.square(y-y_pred))/np.sum(np.square(y - np.average(y_pred, 0)), 0))
def accuracy(y, y_pred):
    '''Calculate the accuracy.
    Can handle multi-class if each class has it's own column.
    
    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The accuracy'''
    if y.ndim != y_pred.ndim:
        raise Exception('Dimensions must match for both arguments.')
    if y.ndim == 1:
        y = np.expand_dims(y, 1)
        y_pred = np.expand_dims(y_pred, 1)
    rows, cols = y.shape
    TP = 0
    for col in range(cols):
        TP += np.sum(y[:,col]==y_pred[:, col])
    return TP/(rows*cols)
def error_rate(y, y_pred):
    '''Calculate the error rate.
    Can handle multi-class if each class has it's own column.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: the error rate'''
    return 1-accuracy(y, y_pred)

def precision(y, y_pred, kind='MICRO'):
    '''Calculate the precision.
    Can handle multi-class if each class has it's own column.
    Kind only matters if multi-class.

    
    :param y: the true values
    :param y_pred: the predicted values
    :param kind: 'MICRO', 'MACRO', or 'WEIGHTED'
    :rtype: int
    :return: The precision'''

    if np.shape(y)[1] == 1:
        return np.sum(y_pred[y == 1] == 1)/np.sum(y_pred == 1)
    match kind.upper():
        case 'MICRO':
            return precision_micro(y, y_pred)
        case 'MACRO':
            return precision_macro(y, y_pred)
        case 'WEIGHTED':
            return precision_weighted(y, y_pred)
        case _:
            raise Exception(f'Invalid Input {kind}')
def precision_micro(y, y_pred):
    '''Calculate the multi-class micro precision.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The precision'''

    rows, cols = np.shape(y)
    TP = 0
    P = 0
    for col in range(cols):
        TP += np.sum(y_pred[y[:,col]==1, col]==1)
        P += np.sum(y_pred[:, col]==1)
    return TP/P

def precision_macro(y, y_pred):
    '''Calculate the multi-class macro precision.
    
    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The precision'''
        
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])
    return total/cols

def precision_weighted(y, y_pred):
    '''Calculate the multi-class weighted precision.
    
    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The precision'''
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def recall(y, y_pred, kind='MICRO'):
    ''''Calculate the recall.
    Can handle multi-class if each class has it's own column.
    Kind only matters if multi-class.

    :param y: the true values
    :param y_pred: the predicted values
    :param kind: 'MICRO', 'MACRO', or 'WEIGHTED'
    :rtype: int
    :return: The recall'''
    if np.shape(y)[1] == 1:
        return np.sum(y_pred[y == 1] == 1)/(np.sum(y_pred[y == 1] == 1) + np.sum(y_pred[y == 1] == 0))
    match kind.upper():
        case 'MICRO':
            return recall_micro(y, y_pred)
        case 'MACRO':
            return recall_macro(y, y_pred)
        case 'WEIGHTED':
            return recall_weighted(y, y_pred)
        case _:
            raise Exception(f'Invalid Input {kind}')

def recall_micro(y, y_pred):
    ''''Calculate the multi-class micro recall.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The recall'''
    rows, cols = np.shape(y)
    TP = 0
    FN = 0
    for col in range(cols):
        TP += np.sum(y_pred[y[:,col]==1, col]==1)
        FN += np.sum(y_pred[y[:,col]==1, col]==0)
    return TP/(TP+FN)

def recall_macro(y, y_pred):
    ''''Calculate the multi-class macro recall.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The recall'''
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])
    return total/cols

def recall_weighted(y, y_pred):
    ''''Calculate the multi-class weighted recall.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The recall'''
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += recall(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def f1(y, y_pred, kind='MICRO'):
    ''''Calculate the f1 score.
    Can handle multi-class if each class has it's own column.
    Kind only matters if multi-class.

    :param y: the true values
    :param y_pred: the predicted values
    :param kind: 'MICRO', 'MACRO', or 'WEIGHTED'
    :rtype: int
    :return: The f1 score'''
    if np.shape(y)[1] == 1:
        return 2*precision(y, y_pred)*recall(y, y_pred)/(precision(y, y_pred) + recall(y, y_pred))
    match kind.upper():
        case 'MICRO':
            return f1_micro(y, y_pred)
        case 'MACRO':
            return recall_macro(y, y_pred)
        case 'WEIGHTED':
            return recall_weighted(y, y_pred)
        case _:
            raise Exception(f'Invalid Input {kind}')
        
def f1_micro(y, y_pred):
    ''''Calculate the multi-class micro f1 score.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The f1 score'''
    return 2*precision_micro(y, y_pred)*recall_micro(y, y_pred)/(precision_micro(y, y_pred) + recall_micro(y, y_pred))

def f1_macro(y, y_pred):
    ''''Calculate the multi-class macro f1 score.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The f1 score'''
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += f1(y[:,col], y_pred[:,col])
    return total/cols

def f1_weighted(y, y_pred):
    ''''Calculate the multi-class weighted f1 score.

    :param y: the true values
    :param y_pred: the predicted values
    :rtype: int
    :return: The f1 score'''
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += f1(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def split(*X, train_split=0.7):
    '''Given any number of inputs, shuffle them and split them according to train_split
    
    :param *X: The inputs to shuffle and split
    :param train_split: decimal in range (0, 1) representing training split. Defaults to 0.7
    :rtype: tuple
    :return: A tuple where each input has been split into two. [i.e. (X, y) -> (X_train, X_test, y_train, y_test)]'''
    rows, _ = np.shape(X[0])
    for x in X:
        if np.shape(x)[0] != rows:
            raise Exception("All inputs must have the same number of rows")

    rng = np.random.default_rng()
    idx = rng.permutation(rows)
    y = list()
    row_split = round(rows*train_split)
    for x in X:
        if x.ndim == 1:
            x = np.expand_dims(x, 1)
        shuffled = x[idx]
        y.append(shuffled[:row_split])
        y.append(shuffled[row_split:])
    return tuple(y)


class Scaler():
    '''A Z-score scaler'''
    def __init__(self) -> None:
        self.average = None
        self.std = None
    def fit(self, X):
        '''Fits the scaler to the data
        
        :param X: the data to fit the scaler on
        :raises Exception: Raised if scaler is already fit
        :rtype: NoneType
        :return: None'''
        if self.average is not None:
            raise Exception("Scaler already fitted")
        self.average = np.average(X, 0)
        self.std = np.std(X, 0)
    def transform(self, X):
        '''Performs Z-score standardization on the data
        
        :param X: the data to perform standardization on
        :rtype: ndarray
        :return: the transformed data'''

        _, cols = X.shape
        return (X - self.average)/self.std
    def fit_transform(self, X):
        '''Fits and then transforms the data
        
        :param X: the data to fit the scaler and perform standardization on
        :raises Exception: Raised if scaler is already fit
        :rtype: ndarray
        :return: the transformed data'''
        self.fit(X)
        return self.transform(X)








