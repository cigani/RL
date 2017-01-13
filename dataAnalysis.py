import numpy as np
import h5py
import os
import glob
import pickle


class Loader:
    def __init__(self):
        self.PATH = os.getcwd()
        self.DATA = glob.glob(self.PATH + "/*.hdf5")
        self.force_data = []
        self.x_data = []
        self.x_d_data = []
        self.episode_rewards = []
        self.data_dict = {}

    def load_data(self):
        print(self.PATH)
        self._start()

    def _start(self):
        for data_file in self.DATA:
            with h5py.File(data_file, 'r') as f:
                for key in f:
                    for val_index, val in enumerate(f['{}'.format(key)]):
                        self.data_dict['{}'.format(val)] = \
                            np.array(f['{}'.format(key)]['{}'.format(val)])

        pickle.dump(self.data_dict, open("data-dict", "wb", ), protocol=2)

    def view_pickle(self):
        ''' Do whatever you need in here this should be a cleaner way to
        view the data set for you though'''
        pickle_view = pickle.load(open("{0}/data-dict".format(self.PATH)),
                                  "rb")



m = Loader()
m.load_data()
