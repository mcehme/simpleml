import numpy as np

class KNN():
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X, k=3):
        y_pred = np.empty((np.size(X, 0), np.size(self.y_train, 1)))
        distances = np.empty((np.size(self.X_train, 0), 1))

        # core algorithm: for every vector go through every vector in the training set and match to k nearest neighbors
        for i, vector in enumerate(X):
            for j, training_vector in enumerate(self.X_train):
                distances[j] = self.__euclidean_dist(vector, training_vector)
            indices = np.argsort(distances, 0)
            labels = self.y_train[indices[0:k]]
            values, count = np.unique(labels, return_counts=True,axis=0)
            max = np.argmax(count, axis=0)
            y_pred[i] = values[max]
        return y_pred

    def __euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    

class LVQ:
    def fit(self, X, y, epochs=1000, alpha=0.1, num_neurons=None) -> None:
        # Get the number of classes from the data and ensure we have an adequate number of neurons
        c = np.unique(y, axis=0).shape[0]
        if num_neurons is None: num_neurons = c
        if c > num_neurons:
            raise Exception("the # of neurons must be >= to the # of classes")
        indices = np.empty((num_neurons), np.int32)
        n_neuron = 0
        classes = list()
        i = 0
        # select 1 from each class
        while len(classes) < c and i < np.size(y, 0):
            if not self.__in(y[i], classes):
                classes.append(y[i])
                indices[n_neuron] = i
                n_neuron += 1
            i += 1
        i = 0
        # for the remaining neurons, select sequentially from  beginning (assume this is random)
        while n_neuron < num_neurons:
            if i not in indices:
                indices[n_neuron] = i
                n_neuron += 1
            i += 1
    

        # separate into neurons and training data
        self.neurons = X[indices]
        self.neurons_labels = y[indices]
        train_data = np.delete(X, indices, 0)
        train_labels = np.delete(y, indices, 0)

        # for the given number of epochs, cycle through the training vectors
        # and update the codebook vector closes to each training vector
        for _ in range(epochs):
            for  vector, label in zip(train_data, train_labels):
                distance = np.empty((self.neurons.shape[0], 1))
                for i, neuron in enumerate(self.neurons):
                    distance[i] = self.__euclidean_dist(vector, neuron)
                i_min = np.argmin(distance)
                if not np.isscalar(i_min): i_min = i_min[0]
                if (self.neurons_labels[i_min] == label).all():
                    self.neurons[i_min,:] = self.neurons[i_min,:] + alpha * (vector - self.neurons[i_min,:])
                else:
                    self.neurons[i_min,:] = self.neurons[i_min,:] - alpha * (vector - self.neurons[i_min,:])

    # cycle through the input vector and test against each codebook vector
    # the closest codebook vector is the winner for that input vector
    def predict(self, X) -> np.ndarray:
        distance = np.empty((self.neurons.shape[0]))
        labels = np.empty((X.shape[0], self.neurons_labels.shape[1]))
        for k, vector in enumerate(X):
            for i, neuron in enumerate(self.neurons):
                distance[i] = self.__euclidean_dist(vector, neuron)
                i_min = np.argmin(distance)
                if not np.isscalar(i_min): i_min = i_min[0]
                labels[k] = np.squeeze(self.neurons_labels[i_min])
        return labels
    def __in(self, vec, lst):
        for element in lst:
            if np.array_equal(element, vec):
                return True
        return False
    def __euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))