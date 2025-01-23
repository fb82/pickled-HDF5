import _pickle as cPickle
import numpy as np
import io
import h5py

# basic interface class to handle pickle objects in hdf5 files
# by default the 'picked' prefix is append to pickled object to distinguish them in the base hdf5 file 
class pickled_hdf5:
    @staticmethod
    def as_numpy(data):
        bf = io.BytesIO()
        cPickle.dump(data, bf)

        buffer = bf.getbuffer()
        return np.frombuffer(buffer, dtype='uint8')


    @staticmethod
    def from_numpy(data):
        m = data
        d = io.BytesIO(m.tobytes())

        return cPickle.load(d)


    def __init__(self, filename, mode='a', label_prefix='pickled'):
        self.hdf5 = h5py.File(filename, mode)
        self.label_prefix = label_prefix


    def get_hdf5(self):
        return self.hdf5


    def get_keys(self):
        keys = []
        def check_item(key, what):
            if isinstance(what, h5py.Dataset): keys.append(what.name)

        self.hdf5[self.label_prefix].visititems(check_item)

        l = len(self.label_prefix)
        return [key[l:] for key in keys]


    def add(self, label, data, overwrite=True, allow_delete_group=False, hdf5_args={'compression':'gzip', 'compression_opts':9}):
        true_label = self.label_prefix + label

        key_exist = self.hdf5.__contains__(true_label)
        if (key_exist):
            if (not overwrite): return False
            if (not isinstance(self.hdf5[true_label], h5py.Dataset)) and (not allow_delete_group): return False
            del self.hdf5[true_label]

        v = pickled_hdf5.as_numpy(data)
        self.hdf5.create_dataset(true_label, data=v, **hdf5_args)
        return True 


    def contain(self, label):
        true_label = self.label_prefix + label
        
        key_exist = self.hdf5.__contains__(true_label)

        if key_exist:
            is_valid = isinstance(self.hdf5[true_label], h5py.Dataset)
        else:
            is_valid = None

        return key_exist, is_valid 
        

    def remove(self, label, allow_delete_group=False):
        true_label = self.label_prefix + label

        key_exist = self.hdf5.__contains__(true_label)

        if key_exist:
            is_valid = isinstance(self.hdf5[true_label], h5py.Dataset)

            if (not is_valid) and (not allow_delete_group):
                return False

            del self.hdf5[true_label]
            return True            
        else:
            return False


    def get(self, label):
        true_label = self.label_prefix + label

        key_exist = self.hdf5.__contains__(true_label)
        if key_exist:
            is_valid = isinstance(self.hdf5[true_label], h5py.Dataset)
            
        if (not key_exist) or (not is_valid):
            return None, False

        return  pickled_hdf5.from_numpy(np.array(self.hdf5[true_label])), True


    def close(self):
        self.hdf5.close()


import torch
if __name__ == '__main__':
    dummy_data_1 = [np.full((3000, 4000), 10), torch.rand([40, 30], device='cuda')]
    print(f"dummy_data_1: {dummy_data_1}")

    dummy_data_2 = {'a': 'nothing', 'b': torch.rand([5, 5], device='cuda')} 
    print(f"dummy_data_2: {dummy_data_2}")

    print("creating pickled-hdf5 database")
    pkh5 = pickled_hdf5('database.hdf5')
    
    print("adding dummy_data_1 as key '/something/a'")
    pkh5.add('/something/a', dummy_data_1)

    print("adding dummy_data_2 as key '/something/b/other'")    
    pkh5.add('/something/b/other', dummy_data_2)

    print("getting hdf5 internal pointer")
    h5 = pkh5.get_hdf5()

    print("adding array [0 1 2 3] as key '/something_else'")
    h5['/something_else'] = np.asarray([0, 1, 2, 3])

    print("closing database")
    pkh5.close()

    print("reload database as read only")
    pkh5 = pickled_hdf5('database.hdf5', 'r')

    print("getting hdf5 internal pointer")    
    h5 = pkh5.get_hdf5()

    l = list(h5.keys())  
    print(f"printing hdf5 root keys: {l} - note that '/pickled' contains pickled-hdf5 keys")
    
    d = h5['/something_else'][()]
    print(f"'/something_else': {d}")
 
    l = list(pkh5.get_keys())
    print(f"all pickled keys: {l}")
    
    t = pkh5.contain('/missed')
    print(f"pickled database contains /'missed': {t}")

    t = pkh5.contain('/something')
    print(f"pickled database contains '/something' which is a hdf5 group: {t}")

    t = pkh5.contain('/something/a')
    print(f"pickled database contains '/something/a', which is a pickled data: {t}")

    restored_dummy_data, ok = pkh5.get('/something/a')    
    print(f"dummy_data_1: {restored_dummy_data} - retrieval is ok: {ok}")

    restored_dummy_data, ok = pkh5.get('/something/b/other')    
    print(f"dummy_data_2: {restored_dummy_data} - retrieval is ok: {ok}")
