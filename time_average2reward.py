import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

average_time = []

list_of_files = os.listdir(os.getcwd())
nFiles = 0

plt.figure('default')
for filename in list_of_files:
    if filename.startswith('default'):
        dataset = []
        print(filename)
        f0 = h5py.File(filename, 'r')

        group = f0['time_to_reward']

        i=0
        while i < np.size(group):
            dataset.append(group['time_to_reward_'+str(i)])
            i +=1

        print(np.size(dataset[:]))
        plt.plot(np.array(dataset[:]), label=filename)

        i=0
        while i < np.size(dataset):
            try:
                average_time[i] += dataset[i][:]
            except IndexError:
                average_time.append(dataset[i][:])
            i += 1

        f0.close()
        nFiles += 1

plt.legend()
plt.xlabel('Episode Number')
plt.ylabel('Time to Reward')
plt.savefig('test.jpg')
plt.show()

print(average_time)

plt.figure('Average Time to Reward per Episode')
plt.title('Average Time to Reward per Episode')
plt.xlabel('Episode Number')
plt.ylabel('Time to Reward')
plt.plot(np.array(average_time)/nFiles)
plt.savefig('average.jpg')
plt.show()
