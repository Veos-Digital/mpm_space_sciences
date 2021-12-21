"""
Based on
https://triangleinequality.wordpress.com/2014/03/31/neural-networks-part-2/
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
sns.set_style("whitegrid")
sns.set_palette("colorblind")


#activation functions
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_prime(x):
    ex = torch.exp(-x)
    return ex / (1 + ex)**2


def identity(x):
    return x


def identity_prime(x):
    return 1


def tanh(x):
    raise ValueError('Please, implement me!')


def tanh_prime(y):
    raise ValueError('Please, implement me!')


def relu(x):
    raise ValueError('Please, implement me!')


def relu_prime(y):
    raise ValueError('Please, implement me!')


def sub(target, prediction):
    return prediction - target


ACTIVATION_DICT = {
    "sigmoid": {"func": sigmoid, "func_prime": sigmoid_prime},
    "relu": {"func": relu, "func_prime": relu_prime},
    "identity": {"func": identity, "func_prime": identity_prime}
}

@dataclass
class Layer:
    num_units: int
    activation_fun: str
    
    def activation(self, prime=False):
        if self.activation_fun is None:
            return
        key = "func_prime" if prime else "func"
        return ACTIVATION_DICT[self.activation_fun][key]
        

class MLP_Regression:
    def __init__(self, architecture, plot=False):
        """
        x_ = input data
        y_ = targets
        architecture is a tuple containing the parameters for every layer:
        architecture = (layer_0 = (number_of_inputs,
                                    activation_function_0,
                                    derivative_activation_function_0)
                            .
                            .
                            .
                        layer_n = (number_of_units,
                                   activation_function_n,
                                   derivative_activation_function_n)
        plot is a boolean (True for plotting the results during training)
        """
        self.plot = plot
        self.n_layers = len(architecture) #the length of the tuple corresponds to the number of layers
        self.layers_size = [layer.num_units for layer in architecture] #number of neurons per layer (excluding biases)
        self.activation_functions = [layer.activation() for layer in architecture] #activation functions per layer
        self.fprimes = [layer.activation(prime=True) for layer in architecture] #derivative of each activation function
        self.build_net() #build the architecture described in architecture

    def build_net(self):
        ### We create lists to store the matrices needed by the chosen architecture
        self.inputs = [] #list of inputs (it contains the input to each layer of the network)
        self.outputs = [] #list of outputs (idem)
        self.W = [] #list of weight matrices (per layer)
        self.b_ = [] #list of bias vectors (per layer)
        self.errors = [] #list of error vectors (per layer)

        ### Initialise weights and biases
        for layer in range(self.n_layers - 1):
            n = self.layers_size[layer]
            m = self.layers_size[layer + 1]
            self.W.append(torch.normal(0, 1, (m, n)))
            self.b_.append(torch.zeros((m, 1)))
            self.inputs.append(torch.zeros((n, 1)))
            self.outputs.append(torch.zeros((n, 1)))
            self.errors.append(torch.zeros((n, 1)))

        n = self.layers_size[-1]
        self.inputs.append(torch.zeros((n, 1)))
        self.outputs.append(torch.zeros((n, 1)))
        self.errors.append(torch.zeros((n, 1)))

    def feedforward(self, x):
        """
        x is propagated through the architecture
        """
        x = torch.unsqueeze(x,1)
        self.inputs[0] = x
        self.outputs[0] = x

        for i in range(1, self.n_layers):
            self.inputs[i] = self.W[i - 1] @ self.outputs[i - 1] + self.b_[i - 1]
            self.outputs[i] = self.activation_functions[i](self.inputs[i])

        return self.outputs[-1]
    
    @staticmethod
    def outer(v1, v2):
        return torch.outer(torch.squeeze(v1,1), torch.squeeze(v2,1))

    def backprop(self, loss_value):
        #Weight matrices and biases are updated based on a single input x and its target y
        self.errors[-1] = self.fprimes[-1](self.outputs[-1]) * loss_value
        n = self.n_layers - 2
        #Again, we will treat the last layer separately
        for i in range(n, 0, -1):
            self.errors[i] = self.fprimes[i](self.inputs[i]) * self.W[i].T @ self.errors[i + 1]
            self.W[i] = self.W[i] - self.learning_rate * self.outer(self.errors[i + 1],self.outputs[i])
            self.b_[i] = self.b_[i] - self.learning_rate * self.errors[i + 1]

        self.W[0] = self.W[0] - self.learning_rate * self.outer(self.errors[1],self.outputs[0])
        self.b_[0] = self.b_[0] - self.learning_rate * self.errors[1]

    def train(self, xs, ys, n_iter, learning_rate = .1, loss=sub):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.learning_rate = learning_rate
        n = xs.shape[0]

        if self.plot:
            self.initialise_figure(xs, ys)

        for repeat in range(n_iter):
            out_xs = []
            
            for row in range(n):    
                x, y = xs[row], ys[row]
                out_x = self.feedforward(x)
                if torch.isnan(out_x):
                    raise ValueError('The model diverged')
                self.backprop(loss(y, out_x))
                out_xs.append(out_x)    
                
            if self.plot and repeat % 100 == 0:
                self.plot_fit_during_train(xs, ys, out_xs)

    def initialise_figure(self, xs, ys):
        plt.ion() #enable interactive plotting
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.gt_plot = self.ax.plot(xs, ys, label='Sine', linewidth=2)
        self.preds_plot, = self.ax.plot(xs, torch.zeros_like(xs), label="Learning Rate: "+str(self.learning_rate))
        plt.legend()
        self.epsilon = 1e-10 # matplotlib needs to pause for a little while in order to display the update of the weights

    def plot_fit_during_train(self, xs, ys, out_xs):
        out_xs = torch.reshape(torch.tensor(out_xs), [-1,1])
        self.preds_plot.set_ydata(out_xs)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(self.epsilon)
        # plt.show(False)
        plt.draw()

    def predict(self, xs):
        n = len(xs)
        m = self.layers_size[-1]
        ret = torch.ones((n,m))

        for i in range(len(xs)):
            ret[i,:] = self.feedforward(xs[i])

        return ret

def test_regression(plots=True):
    pass
    #Create data


if __name__ == "__main__":
    # test_regression()
    n = 20
    xs = torch.linspace(0, 3 * torch.pi, steps=n)
    xs = torch.unsqueeze(xs, 1)
    ys = torch.sin(xs) + 1
    n_iter = 3000
    #set the architecture
    param = (Layer(1, None),
             Layer(20, "sigmoid"),
             Layer(1, "identity"))
    #Set learning rate.
    lrs = [0.3]
    # predictions = []
    for learning_rate in lrs:
        model = MLP_Regression(param, plot=True)
        model.train(xs, ys, n_iter, learning_rate = learning_rate)
        # pred = model.predict(X)
        # print('pred = ', pred)
        # predictions.append([learning_rate, pred])
