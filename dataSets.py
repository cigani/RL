import h5py


def generate_data_save(h5data, _, time_to_reward_data, steps_to_reward_data,
                       weights_data):
    h5data['time_to_reward'].create_dataset(
        "time_to_reward_{}".format(_), data=time_to_reward_data,
        compression="gzip")
    h5data['steps_to_reward'].create_dataset(
        "steps_to_reward_{}".format(_), data=steps_to_reward_data,
        compression="gzip")
    h5data['weights'].create_dataset(
        "weights_{}".format(_), data=weights_data, compression="gzip")


def generate_data_sets(filename, centers):
    h5data = h5py.File(filename, 'w')
    h5data.create_group('time_to_reward')
    h5data.create_group('steps_to_reward')
    h5data.create_group('neurons')
    h5data.create_group('weights')

    h5data['neurons'].create_dataset("centers", data=centers,
                                     compression="gzip")
    return h5data
