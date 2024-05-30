import json
import io
import os.path
import time
import h5py


def dict_to_json(data, path='cwd'):
    # Save Data Test
    # with open('data.json', 'w') as f:
    #     f.write(json.dump(data, f))
    if path =='cwd':
        raise NotImplementedError


    to_unicode = str
    data['Json_filename'] = str(time.strftime("%Y-%m-%d_H%H-M%M"))
    # Write JSON file
    with io.open(os.path.join(path,'{}.json'.format(time.strftime("%Y-%m-%d_H%H-M%M"))), 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    print('Saving is done Done')

def create_hdf5(data,name, path):
    if path == 'cwd':
        raise NotImplementedError
    hf = h5py.File(os.path.join(path,'array_data.h5'), 'w')
    hf.create_dataset(name, data=data)
    hf.close()

def add_nparray_to_hdf5(data, name, path):
    if path == 'cwd':
        raise NotImplementedError
    hf = h5py.File(os.path.join(path, 'array_data.h5'), 'a')
    hf.create_dataset(name, data=data)
    hf.close()
    pass
