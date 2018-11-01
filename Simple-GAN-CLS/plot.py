import numpy as np
import matplotlib.pyplot as plt

with open("graph-without-600iter.txt") as f:
    data = f.read()

data = data.split('\n')

errD = [row.split(' ')[1] for row in data]
errG = [row.split(' ')[2] for row in data]
errRNN = [row.split(' ')[3] for row in data]
iterations = [row.split(' ')[0] for row in data]

fig = plt.figure()

plt.plot(iterations, errD, label = "Discriminator") 
plt.plot(iterations, errG, label = "Generator") 
plt.plot(iterations, errRNN, label = "RNN") 

ax1 = fig.add_subplot(111)

ax1.set_title("Iter vs loss")    
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Losses')

leg = ax1.legend()

plt.show()