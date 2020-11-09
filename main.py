import nn
import numpy as np
from itertools import count

# wfn = weight file name. where the weights are saved/loaded
wfn = "weights.npy"

try:
	with open(wfn, "rb") as f:
		weights = np.load(f)
except:
	weights = np.array([[.1],
											[.2]])

tinputs = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])

toutputs = np.array([[1],[1],[1],[0]])


net = nn.Net(wfn, weights, tinputs, toutputs)

print(net)

for i in count(0):
	net.train()
	in1 = net.run(np.array([1, 1]))
	in2 = net.run(np.array([0, 1]))
	in3 = net.run(np.array([1, 0]))
	in4 = net.run(np.array([0, 0]))


	print(f"\n--------------------\n\nNut: {i+1}\n\n{in1}\n\n{in2}\n\n{in3}\n\n{in4}\n")
