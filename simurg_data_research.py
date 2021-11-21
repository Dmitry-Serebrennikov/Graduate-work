import h5py
import numpy as np

with h5py.File('dtec_20_60_2017_148_-90_90_N_-180_180_E_6c18.h5', 'r') as hdf:
    #structure_items = list(hdf.items())
    #print('Structure items: \n', structure_items)
    group_list = list(hdf.keys())
    print(group_list)
    data = hdf.get('data')
    print(data)
    #dataset_array = np.array(data)
    #print(dataset_array)
    dataset = np.array(data.get('2017-05-28 23:45:00.000000'))
    print(dataset)