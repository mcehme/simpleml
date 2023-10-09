import numpy as np

class FFNeuron:
    def __init__(self, num_inputs)-> None:
        self.W = np.random.rand(num_inputs + 1, 1)

        
    def h(self, x):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        if x.ndim < 2:
            x = np.expand_dims(x, 1)
        return sigmoid(x@self.W)

class FFLayer:
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.compiled = False
        self.num_inputs = num_inputs
    def compile(self):
        if self.compiled:
            raise Exception("Cannot compile an already compiled layer")
        self.neurons = list()
        for _ in range(self.num_neurons):
            self.neurons.append(FFNeuron(self.num_inputs))
        self.compiled = True
        return self.num_neurons
    def feed_forward(self, X):
        self.output = np.zeros((np.shape(X)[0], self.num_neurons))
        for i, neuron in enumerate(self.neurons):
            self.output[:,i] = np.squeeze(neuron.h(X))
        return self.output
    
    def back_propagate(self, h, X, alpha, upstream_diff=None, y=None):
        sigma_gradient = lambda y: y*(1-y)

        if y is not None and y.ndim < 2:
            y = np.expand_dims(y, 1)
        
        diff = np.zeros((np.shape(h)[0], self.num_inputs+1))
        if upstream_diff is None:
            for i, neuron in enumerate(self.neurons):
                int_diff = h[:,i+1] - y[:,i]
                diff += np.expand_dims(int_diff, 1) * np.expand_dims(sigma_gradient(h[:,i+1]), 1)*np.transpose(neuron.W)
                final_gradient = int_diff * sigma_gradient(h[:,i+1])
                final_gradient = np.expand_dims(final_gradient, 1)
                final_gradient = np.transpose(final_gradient)@X
                final_gradient = 1/np.shape(final_gradient)[0] * np.sum(final_gradient, 0)
                final_gradient = np.expand_dims(final_gradient, 1)
                neuron.W -= alpha*final_gradient
        else:
            for i, neuron in enumerate(self.neurons):
                diff += np.expand_dims(upstream_diff[:, i+1], 1)*np.expand_dims(sigma_gradient(h[:,i+1]), 1)*np.transpose(neuron.W)
                final_gradient = upstream_diff[:, i+1] * sigma_gradient(h[:,i+1])
                final_gradient = np.expand_dims(final_gradient, 1)
                final_gradient = np.transpose(final_gradient)@X
                final_gradient = 1/np.shape(final_gradient)[0] * np.sum(final_gradient, 0)
                final_gradient = np.expand_dims(final_gradient, 1)
                neuron.W -= alpha*final_gradient
        return diff

class FFNeuralNet:
    def __init__(self):
        self.layers = list()
        self.compiled = False
    def add_layer(self, layer):
        if self.compiled:
            raise Exception("Cannot add layer to compiled model")
        self.layers.append(layer)
    def compile(self):
        for layer in self.layers:
            layer.compile()
        self.compiled = True
    def fit(self, X, y, alpha=0.05, epochs=1000):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        for _ in range(epochs):
            outputs_layers = self.feed_forward(X)
            self.back_propagate(outputs_layers, y, alpha)
    def feed_forward(self, X):
        output = list()
        output.append(X)
        for i, layer in enumerate(self.layers):
            X = layer.feed_forward(X)
            m = np.size(X, 0)
            X = np.hstack((np.ones((m, 1)), X))
            output.append(X)
        return output
    def back_propagate(self, outputs_layers, y, alpha):
        upper_diff = None
        diff_list = list()
        for i, layer in enumerate(reversed(self.layers)):
            output_layer = outputs_layers[(- 1 - i)]
            inner_layer = outputs_layers[(- 1 - i - 1)]
            diff_list.append(layer.back_propagate(output_layer, inner_layer, alpha,  upper_diff, y))
            upper_diff = diff_list[-1]
            y = None

    def predict(self, X):
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        for i, layer in enumerate(self.layers):
            X = layer.feed_forward(X)
            m = np.size(X, 0)
            X = np.hstack((np.ones((m, 1)), X))
        return np.round(X[:, 1:])