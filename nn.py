import numpy as np

# activation function. used to get output
sigmoid = lambda x : 1/(1+np.exp(-x))
# derivitive of the activation function. used to change weights
sigder = lambda x : sigmoid(x)*(1-sigmoid(x))

class Net:
	# runs when a new Net is created
	def __init__(self, wfn, weights, tinputs, toutputs, trainlen=30000, bias=.3, lr=.05):
		self.wfn = wfn
		self.weights = weights
		self.tinputs = tinputs
		self.toutputs = toutputs
		self.trainlen = trainlen
		self.bias = bias
		self.lr = lr

	# runs when a Net is interracted with eg (print(Net))
	def __repr__(self):
		return f"{self.weights}"


	# training function
	def train(self):
		for j in range(self.trainlen):
			self.inputs = self.tinputs

			# takes the sigmoid of the dot product (matrix multiplication) of the inputs and weights
			self.inp = np.dot(self.inputs, self.weights)
			self.out = sigmoid(self.inp)

			# calculates how wrong the guess was
			self.error = self.out - self.toutputs

			# calculates the derivitive of the function (inputs -> outputs)
			self.derror_dout = self.error
			self.dout_din = sigder(self.out)
			self.deriv = self.derror_dout * self.dout_din
			# transposes the inputs
			self.inputs = self.tinputs.T
			self.derivfin = np.dot(self.inputs, self.deriv)

			# finally, update the weights and the bias
			self.weights -= self.lr * self.derivfin
			for i in self.deriv:
				self.bias -= i * self.derivfin
		
		# save the updated weights
		with open(self.wfn, "wb") as f:
			np.save(f, self.weights)

	def run(self, point):
		return sigmoid(np.dot(point, self.weights))
		
