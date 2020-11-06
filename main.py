import nn
import numpy as np

# wfn = weight file name. where the weights are saved/loaded
wfn = "weights.npy"

try:
	with open(wfn, "rb") as f:
		weights = np.load(f)
except:
	weights = np.array([[.1],
											[.2]])

tinputs = np.array([[1, 1],[1, 0],[0, 1],[0, 0],[0, 0],[0, 0]])

toutputs = np.array([[1],[1],[1],[0],[0],[0]])


net = nn.Net(wfn, weights, tinputs, toutputs)

print(net)


net.train()

print(net.run(np.array([1, 1])))
print(net.run(np.array([1, 0])))
print(net.run(np.array([0, 1])))
print(net.run(np.array([0, 0])))