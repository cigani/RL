import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

filename = "test-01-17-18.58.34s.hdf5"
runtype = "test"

dataset = []
print filename
f0 = h5py.File(filename, 'r')

group = f0['time_to_reward']

i=0
while ( i < np.size(group) ):
    dataset.append(group['time_to_reward_'+str(i)])
    i +=1

print np.size(dataset[:])

plt.figure('Time to Reward per Episode '+str(runtype))
plt.title('Time to Reward per Episode '+str(runtype))
plt.xlabel('Episode Number')
plt.ylabel('Time to Reward')
plt.plot(np.array(dataset[:]))
plt.savefig(str(runtype)+'.jpg')
plt.show()
