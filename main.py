import numpy as np
import nn as nut
from itertools import count

wfn0 = "weights0.npy"
wfn1 = "weights1.npy"

tinputs = np.array([[1, 1],
                    [1, 0],
                    [0, 1],
                    [0, 0]])

toutputs0 = np.array([[1],[1],[1],[0]])
toutputs1 = np.array([[0],[0],[0],[1]])

try:
    with open(wfn0, "rb") as f:
        weights0 = np.load(f)
except:
    weights0 = np.array([[.1],[.2]])

try:
    with open(wfn1, "rb") as f:
        weights1 = np.load(f)
except:
    weights1 = np.array([[.1],[.2]])


net0 = nut.NN(tinputs, toutputs0, weights0, wfn0)
net1 = nut.NN(tinputs, toutputs1, weights1, wfn1)


for a in count(0):
    net0.train()
    net1.train()

    in1 = np.array([1, 1])
    in2 = np.array([1, 0])
    in3 = np.array([0, 1])
    in4 = np.array([0, 0])

    out1 = np.array([net1.run(in1), net0.run(in1)])
    out2 = np.array([net1.run(in2), net0.run(in2)])
    out3 = np.array([net1.run(in3), net0.run(in3)])
    out4 = np.array([net1.run(in4), net0.run(in4)])    
    
    print(f"\n--------------------\n\ngen: {a+1}\n\n{out1}\n\n{out2}\n\n{out3}\n\n{out4}")