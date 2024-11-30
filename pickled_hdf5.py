import _pickle as cPickle
import numpy as np
import io
import h5py

# basic interface class to handle pickle objects in hdf5 files
# by default the '/picked' prefix is append to pickled object to distinguish them in the base hdf5 file 
class pickled_hdf5:
    def __init__(self, filename, mode='a', label_prefix='/pickled'):
        self.hdf5 = h5py.File(filename, mode)
        self.label_prefix = label_prefix


    def get_hdf5(self):
        return self.hdf5


    def get_keys(self):
        return self.hdf5[self.label_prefix].keys()


    def add(self, label, data, overwrite=True, hdf5_args={'compression':'gzip', 'compression_opts':9}):
        true_label = self.label_prefix + label

        bf = io.BytesIO()
        cPickle.dump(data, bf)

        buffer = bf.getbuffer()
        v = np.frombuffer(buffer, dtype='uint8')
        
        if true_label in self.hdf5.keys():
            if overwrite:
                del self.hdf5[true_label]
            else:
                return False

        self.hdf5.create_dataset(true_label, data=v, **hdf5_args)
        return True


    def contain(self, label):
        if (self.label_prefix + label) in self.hdf5.keys(): return True
        else: return False


    def remove(self, label):
        true_label = self.label_prefix + label

        if true_label in self.hdf5.keys():
            del self.hdf5[true_label]
            return True
        else:
            return False


    def get(self, label):
        true_label = self.label_prefix + label

        if not(true_label in self.hdf5.keys()):
            return None, False

        m = np.array(self.hdf5[true_label])
        d = io.BytesIO(m.tobytes())

        return cPickle.load(d), True


    def close(self):
        self.hdf5.close()


import torch
if __name__ == '__main__':
    dummy_data = [np.full((3000, 4000), 10), torch.rand([40, 30], device='cuda')]
    print(dummy_data)

    pkh5 = pickled_hdf5('database.hdf5')
    pkh5.add('/something', dummy_data)
    h5 = pkh5.get_hdf5()
    h5['/something_else'] = np.asarray([0, 1, 2, 3])
    pkh5.close()

    pkh5 = pickled_hdf5('database.hdf5', 'r')
    
    h5 = pkh5.get_hdf5()
    print(list(h5.keys()))
    print(h5['/something_else'][()])
    
    print(list(pkh5.get_keys()))
    print(pkh5.contain('/missed'))
    print(pkh5.contain('/something'))

    restored_dummy_data, ok = pkh5.get('/something')
    print(restored_dummy_data, ok)
