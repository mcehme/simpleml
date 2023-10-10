import numpy as np

class FFNeuron:
    '''A basic neuron for a Feed Forward Layer
    
    :param num_inputs: The number of inputs to the neuron (determines the number of weights)
    :rtype: NoneType
    :return: None'''

    def __init__(self, num_inputs)-> None:
        self.W = np.random.rand(num_inputs + 1, 1)

        
    def __h(self, x):
        '''Creates the neurons.

        Name mangling to help ensure only called by FFLayer

        :param x: The input to the neuron
        :rtype: ndarray
        :return: The output of the neuron'''

        sigmoid = lambda x: 1/(1 + np.exp(-x))
        if x.ndim < 2:
            x = np.expand_dims(x, 1)
        return sigmoid(x@self.W)

class FFLayer:
    '''A basic Layer for a Feed Forward Neural Net
    
    :param num_neurons: The number of neurons in the layer
    :param num_inputs: The number of inputs for each neuron in the layer
    :rtype: NoneType
    :return: None'''
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.compiled = False
        self.num_inputs = num_inputs
    def __compile(self):
        '''Creates the neurons.
        Name mangling to help ensure only called by FFNeuralNet

        :raises Exception: Raised when attempting to compile already compiled layer
        :rtype: int
        :return: The number of neurons in the layer'''
        if self.compiled:
            raise Exception("Cannot compile an already compiled layer")
        self.neurons = list()
        for _ in range(self.num_neurons):
            self.neurons.append(FFNeuron(self.num_inputs))
        self.compiled = True
        return self.num_neurons
    def __feed_forward(self, X):
        ''''Used to calculated the output for the layer.
        Name mangling to help ensure only called by FFNeuralNet
        
        :param X: The data input
        :rtype: ndarray
        :return: An numpy ndarray representing the combined outputs of the neurons'''
        self.output = np.zeros((np.shape(X)[0], self.num_neurons))
        for i, neuron in enumerate(self.neurons):
            self.output[:,i] = np.squeeze(neuron._FFNeuron__h(X))
        return self.output
    
    def __back_propagate(self, h, X, alpha, upstream_diff=None, y=None):
        '''Combination of back propagation and gradient descent.
       Name mangling to help ensure only called by FFNeuralNet

        :param h: the outputs of the neurons
        :param X: the inputs to the neurons
        :param alpha: the learning rate
        :param upstream_diff: The gradient of the upstream layer. None if the current layer is the output layer
        :param y: The desired output classes. Always None unless the current layer is the output layer.
        :rtype: NoneType
        :return: None'''

        #High Level approach:
        #For each neuron take the upstream gradient and calculate the gradient for the neuron given the below formula
        # gradient = upstream_gradient * h(x)(1-h(x))*x where x is the neuron input and h(x) is the neuron output
        # for simplicity h(x) is passed as the value h to the function

        #For the output layer
        #Upstream gradient is the diff between output and expected (h - y)
        
        #In general, upstream gradient is calculated by the previous layer as follows
        # for each neuron we add it's component to the upstream gradient
        # upstream gradient += upstream gradient * h(x)(1-h(x))*w where h(x) is the neuron output and w is the neuron weight
        # note that each weight corresponds to a different downstream neuron

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
    '''Creates a Feed Forward Neural Net'''
    def __init__(self):
        self.layers = list()
        self.compiled = False
    def add_layer(self, layer):
        '''Adds a FFLayer to the Neural Net

        :param layer: a FFLayer
        :raises Exception: Raised when attempting to add layer to compiled model
        :rtype: NoneType
        :return: None'''
        if self.compiled:
            raise Exception("Cannot add layer to compiled model")
        self.layers.append(layer)
    def compile(self):
        '''Compiles the Neural Net (Tells the the layers to generate neurons)

        :raises Exception: Raised when attempting to compile to compiled model
        :rtype: NoneType
        :return: None
        '''
        if self.compiled:
            raise Exception("Cannot compile already compiled model")
        for layer in self.layers:
            layer._FFLayer__compile()
        self.compiled = True

    def fit(self, X, y, alpha=0.05, epochs=1000):
        '''Fits the neural net to the given training data
        
        :param X: The input data 
        :param y: The target classes
        :param alpha: The learning rate (defaults to 0.05)
        :param epochs: Number of epochs to train for (defaults to 1000)
        :raises Exception: Raised if model is not compiled
        :rtype: NoneType
        :return: None'''

        if not self.compiled:
            raise Exception("Model must be compiled")
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        for _ in range(epochs):
            outputs_layers = self.__feed_forward(X)
            self.__back_propagate(outputs_layers, y, alpha)

    def __feed_forward(self, X):
        '''Propagates the data values through the layers of the neural network
        
        :param X: The input data:
        :rtype: list
        :return: A list of the outputs of each layer'''
        output = list()
        output.append(X)
        for i, layer in enumerate(self.layers):
            X = layer._FFLayer__feed_forward(X)
            m = np.size(X, 0)
            X = np.hstack((np.ones((m, 1)), X))
            output.append(X)
        return output
    
    def __back_propagate(self, outputs_layers, y, alpha):
        '''Takes the results of the feed forward stage and performs back propagation + gradient descent
        
        :param outputs_layers: the outputs of each layer
        :param y: the target classes (the desired outputs)
        :param alpha: the learning rate
        :rtype: NoneType
        :return: None'''
        upper_diff = None
        diff_list = list()
        for i, layer in enumerate(reversed(self.layers)):
            output_layer = outputs_layers[(- 1 - i)]
            inner_layer = outputs_layers[(- 1 - i - 1)]
            diff_list.append(layer._FFLayer__back_propagate(output_layer, inner_layer, alpha,  upper_diff, y))
            upper_diff = diff_list[-1]
            y = None

    def predict(self, X):
        '''Takes an input and computes expected output
        
        :param X: the input data
        :rtype: ndarray
        :return: A numpy ndarry of the output classes'''
        m = np.size(X, 0)
        X = np.hstack((np.ones((m, 1)), X))
        for i, layer in enumerate(self.layers):
            X = layer._FFLayer__feed_forward(X)
            m = np.size(X, 0)
            X = np.hstack((np.ones((m, 1)), X))
        return np.round(X[:, 1:])