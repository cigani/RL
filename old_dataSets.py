import h5py
import numpy as np


def generate_data_save(_, x_data, x_dot_data, force_data, q_data, time_to_reward_data, weights_data):
    h5data['x_data'].create_dataset(
        'x_data_{}'.format(_), data=x_data, compression="gzip")
    h5data['x_dot_data'].create_dataset(
        "x_dot_data_{}".format(_), data=x_dot_data,
        compression="gzip")
    h5data['force_data'].create_dataset(
        "force_data_{}".format(_), data=force_data,
        compression="gzip")
    h5data['q_data'].create_dataset(
        "q_data_{}".format(_), data=q_data, compression="gzip")
    h5data['time_to_reward'].create_dataset(
        "time_to_reward_{}".format(_), data=time_to_reward_data, compression="gzip")
    h5data['weights'].create_dataset(
        "weights_{}".format(_), data=weights_data, compression="gzip")

def generate_data_fields():
    x_data = []
    x_dot_data = []
    force_data = []
    q_data = []
    return force_data, q_data, x_data, x_dot_data

def generate_data_sets(filename):
    h5data = h5py.File(filename, 'w')
    h5data.create_group('time_to_reward')
    h5data.create_group('x_data')
    h5data.create_group('x_dot_data')
    h5data.create_group('force_data')
    h5data.create_group('q_data')
    h5data.create_group('neurons')
    h5data.create_group('weights')

    h5data['neurons'].create_dataset("centers", data=self.centers, compression="gzip")
    return h5data

def generate_data_vectors(force_data, q_data, x_data, x_dot_data,
                          x, x_d, f, hold):
    x_data = np.append(x_data, x)
    x_dot_data = np.append(x_dot_data, x_d)
    force_data = np.append(force_data, f)
    q_data = np.append(q_data, hold)
    return force_data, q_data, x_data, x_dot_data
