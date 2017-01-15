import h5py
import numpy as np


def generate_data_save(episode_count, episode_rewards, force_data, h5data,
                       step_count, q_data, x_data, x_dot_data):
    episode_rewards = np.append(episode_rewards, step_count)
    h5data['x_data'].create_dataset(
        'x_data_{}'.format(episode_count), data=x_data, compression="gzip")
    h5data['x_dot_data'].create_dataset(
        "x_dot_data_{}".format(episode_count), data=x_dot_data,
        compression="gzip")
    h5data['force_data'].create_dataset(
        "force_data_{}".format(episode_count), data=force_data,
        compression="gzip")
    h5data['q_data'].create_dataset(
        "q_data_{}".format(episode_count), data=q_data, compression="gzip")
    return episode_rewards


def generate_data_fields():
    x_data = []
    x_dot_data = []
    force_data = []
    q_data = []
    return force_data, q_data, x_data, x_dot_data


def generate_data_sets(filename):
    h5data = h5py.File(filename, 'w')
    h5data.create_group('episode_rewards')
    h5data.create_group('x_data')
    h5data.create_group('x_dot_data')
    h5data.create_group('force_data')
    h5data.create_group('q_data')
    return h5data
