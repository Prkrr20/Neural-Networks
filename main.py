import nn
import numpy as np
from itertools import count

# wfn = weight file name. where the weights are saved/loaded
wfn1 = "weights1.npy"
wfn2 = "weights2.npy"
wfn3 = "weights3.npy"

try:
	with open(wfn1, "rb") as f:
		weights1 = np.load(f)
except:
	weights1 = np.array([[.1],
										 	 [.2],
											 [.3]])

try:
	with open(wfn2, "rb") as f:
		weights2 = np.load(f)
except:
	weights2 = np.array([[.1],
										 	 [.2],
											 [.3]])

try:
	with open(wfn3, "rb") as f:
		weights3 = np.load(f)
except:
	weights3 = np.array([[.1],
										 	 [.2],
											 [.3]])


tinputs = np.array([[0, 0, 0],
										[0, 1, 0],
										[0, 2, 0],
										[1, 0, 0],
										[1, 1, 0],
										[1, 2, 0],
										[2, 0, 0],
										[2, 1, 0],
										[2, 2, 0]])

toutputs1 = np.array([[1],[0],[0],[1],[0],[0],[1],[0],[0]])
toutputs2 = np.array([[0],[1],[0],[0],[1],[0],[0],[1],[0]])
toutputs3 = np.array([[0],[0],[1],[0],[0],[1],[0],[0],[1]])


net1 = nn.Net(wfn1, weights1, tinputs, toutputs1)
net2 = nn.Net(wfn2, weights2, tinputs, toutputs2)
net3 = nn.Net(wfn3, weights3, tinputs, toutputs3)

print(net1)
print("\n")
print(net2)
print("\n")
print(net3)
print("\n")


for i in count(0):
	net1.train()
	net2.train()
	net3.train()

	in1 = np.array([0, 2, 0]) # 0, 0, 1
	in2 = np.array([0, 2, 0]) # 0, 0, 1
	in3 = np.array([0, 0, 0]) # 1, 0, 0
	in4 = np.array([0, 1, 0]) # 0, 1, 0
	in5 = np.array([0, 0, 0]) # 1, 0, 0

	out1 = np.array([net1.run(in1),net2.run(in1),net3.run(in1)])

	out2 = np.array([net1.run(in2),net2.run(in2),net3.run(in2)])

	out3 = np.array([net1.run(in3),net2.run(in3),net3.run(in3)])

	out4 = np.array([net1.run(in4),net2.run(in4),net3.run(in4)])

	out5 = np.array([net1.run(in5),net2.run(in5),net3.run(in5)])

	print(f"\n---------------\n\nNut: {i+1}\n\nNet 1:\n\n{out1}\n\n{out2}\n\n{out3}\n\nNet 2:\n\n{out4}\n\n{out5}\n")