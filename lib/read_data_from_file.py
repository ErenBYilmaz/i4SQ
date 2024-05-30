import json
import io
import os.path
import time
import h5py
import numpy as np

def read_hdf5(list_of_keys, filename, path='cwd'):
    '''

    :param list_of_keys:
    :param filename:
    :param path:
    :return:

    Name Format: '[...].h5'
    '''
    if path =='cwd':
        raise NotImplementedError
    if list_of_keys==[]:
        raise NotImplementedError
    hf = h5py.File(os.path.join(path,'array_data.h5'))
    data_dict = {}
    for each_key in list_of_keys:
        data_dict[each_key] = hf.get(each_key)
    return data_dict

def get_list_of_keys_from_json(json_name, path):
    with open(os.path.join(path,'{}.json'.format(json_name))) as json_file:
        data = json.load(json_file)
    return data
