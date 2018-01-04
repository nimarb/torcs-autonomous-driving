import pickle
import numpy as np
import argparse

LEARN_DATA = 'learn_data/'
PROCESSED_DATA = 'data/processed_data/'

MAP_NAMES = ('cg_speedway_1', 'brondelhach', 'alpine_1')


def load_file(path):
    return np.load(path)
    # objects = []
    # with (open(path, "rb")) as openfile:
    #     while True:
    #         try:
    #             objects.append(pickle.load(openfile))
    #         except EOFError:
    #             break
    # return objects[0]


def get_label_ind(label):
    if label == 'accel':
        ind = 0
    elif label == 'brake':
        ind = 1
    elif label == 'steering':
        ind = 2
    elif label == 'clutch':
        ind = 3
    else:
        raise NameError('wrong label')
    return ind


def get_inputs(dims=None):
    data = []
    for m in MAP_NAMES:
        data.append(load_file(PROCESSED_DATA + m + '_input.npy'))
    data = np.vstack(data)
    if dims is None:
        return data
    else:
        return data[:, dims]


def get_outputs(label):
    ind = get_label_ind(label)
    data = []
    for m in MAP_NAMES:
        data.append(load_file(PROCESSED_DATA + m + '_output.npy'))
    data = np.vstack(data)
    return np.reshape(data[:, ind], (data.shape[0], 1))
