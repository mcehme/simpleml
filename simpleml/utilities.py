import numpy as np
def mean_squared_error(y, y_pred):
    return np.sum(np.square(y - y_pred))/np.size(y,0)
def r2(y, y_pred):
    return np.squeeze(1 - np.sum(np.square(y-y_pred))/np.sum(np.square(y - np.average(y_pred, 0)), 0))
def accuracy(y, y_pred):
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
    return 1-accuracy(y, y_pred)
def precision(y, y_pred, kind='MICRO'):
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
    rows, cols = np.shape(y)
    TP = 0
    P = 0
    for col in range(cols):
        TP += np.sum(y_pred[y[:,col]==1, col]==1)
        P += np.sum(y_pred[:, col]==1)
    return TP/P

def precision_macro(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])
    return total/cols

def precision_weighted(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def recall(y, y_pred, kind='MICRO'):
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
    rows, cols = np.shape(y)
    TP = 0
    FN = 0
    for col in range(cols):
        TP += np.sum(y_pred[y[:,col]==1, col]==1)
        FN += np.sum(y_pred[y[:,col]==1, col]==0)
    return TP/(TP+FN)

def recall_macro(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += precision(y[:,col], y_pred[:,col])
    return total/cols

def recall_weighted(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += recall(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def f1(y, y_pred, kind='MICRO'):
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
    return 2*precision_micro(y, y_pred)*recall_micro(y, y_pred)/(precision_micro(y, y_pred) + recall_micro(y, y_pred))

def f1_macro(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += f1(y[:,col], y_pred[:,col])
    return total/cols

def f1_weighted(y, y_pred):
    rows, cols = np.shape(y)
    total = 0
    for col in range(cols):
        total += f1(y[:,col], y_pred[:,col])*np.sum(y[:,col]==1)/rows
    return

def split(*X, train_split=0.7):

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
    def __init__(self) -> None:
        self.average = None
        self.std = None
    def fit(self, X):
        if self.average is not None:
            raise Exception("Scaler already fitted")
        self.average = np.average(X, 0)
        self.std = np.std(X, 0)
    def transform(self, X):
        _, cols = X.shape
        return (X - self.average)/self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)








