import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigder(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, tinputs, toutputs, weights, wfn, trainlen=30000, bias=.3, lr=.05):
        self.tinputs = tinputs
        self.toutputs = toutputs
        self.weights = weights
        self.wfn = wfn
        self.bias = bias
        self.lr = lr
        self.trainlen = trainlen
    
    def __repr__(self):
        return f"{self.weights}"


    def train(self):
        for a in range(self.trainlen):
            self.inputs = self.tinputs

            self.inp = np.dot(self.inputs, self.weights)
            self.out = sigmoid(self.inp)

            self.error = self.out - self.toutputs
            self.derror_dout = self.error
            self.dout_din = sigder(self.out)
            self.deriv = self.derror_dout * self.dout_din

            self.inputs = self.tinputs.T
            self.derivfin = np.dot(self.inputs, self.deriv)

            self.weights -= self.derivfin * self.lr

            for b in self.deriv:
                self.bias -= b * self.lr
        with open(self.wfn, "wb") as f:
            np.save(f, self.weights)

    def run(self, point):
        return sigmoid(np.dot(point, self.weights))
