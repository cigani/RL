import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def input_layer(input_center, neuron_center, x_sigma, x_d_sigma):
    rj = np.exp((-(input_center[0] - neuron_center[0]) ** 2) /
                x_sigma - ((input_center[1] - neuron_center[1]) ** 2) /
                x_d_sigma)
    return rj


episode=0
N=10
neuronCount=N*N

_x_space_, x_centers_distance = np.linspace(-150, 30, N,
                                                 retstep=True)
_x_d_space_, phi_centers_distance = np.linspace(-15, 15, N,
                                                     retstep=True)
x_sigma = x_centers_distance ** 2
x_d_sigma = phi_centers_distance ** 2

x_direction = np.zeros(neuronCount)
phi_direction = np.zeros(neuronCount)

x_vector = np.zeros((N,N))
phi_vector = np.zeros((N,N))

Q_val = np.zeros((neuronCount,3))

plt.figure('default')
dataset_centers = []
dataset_weights = []
filename = "vector-01-16-14.01.34s.hdf5"
#filename = "vector-01-16-17.04.01s.hdf5"
print filename
f0 = h5py.File(filename, 'r')

group0 = f0['neurons']
group1 = f0['weights']

dataset_centers.append(np.array(group0['centers']))

i=0
while ( i < np.size(group1) ):
    dataset_weights.append(group1['weights_'+str(i)])
    i +=1

#print dataset_centers[0][0]
#print dataset_weights[1][0][1]

for k in range(3):
    for neuronIndex in range(neuronCount):
        for nodeIndex in range(neuronCount):
            Q_val[neuronIndex,k] \
                += \
                dataset_weights[episode][neuronIndex][k] \
                * input_layer(dataset_centers[0][nodeIndex], \
                              dataset_centers[0][neuronIndex], \
                              x_sigma, \
                              x_d_sigma)


sorted_x = []
i=0
while ( i < neuronCount ):
    sorted_x.append(dataset_centers[0][i][0])
    i += 1

print np.transpose(np.reshape(sorted_x, (N,N)))

sorted_phi = []
i=0
while ( i < neuronCount ):
    sorted_phi.append(dataset_centers[0][i][1])
    i += 1

print np.transpose(np.reshape(sorted_phi, (N,N)))
    
actions = np.argmax(Q_val[:,:],axis=1)

x_direction[actions==0] = 1.
x_direction[actions==1] = -1.
x_direction[actions==2] = 0.

phi_direction[actions==0] = 0.
phi_direction[actions==1] = 0.
phi_direction[actions==2] = 0.

x_vector = np.reshape(x_direction, (N, N))
phi_vector = np.reshape(phi_direction, (N,N))

f0.close()

plt.title('Navigation Map Episode '+str(episode+1))
plt.quiver(np.transpose(x_vector), np.transpose(phi_vector), scale=50)
plt.xlabel('Neuron (Position)')
plt.ylabel('Neuron (Velocity)')
plt.axis([-1, N, -1 , N])  
plt.savefig('naviMap'+str(episode+1)+'.jpg')
plt.show()
