# pickled HDF5
Basic Python interface class to handle pickle objects in hdf5 files, including PyTorch tensors, enabling also compression.

This is achieved by converting Pickle object as byte stream into NumPy byte arrays. By default the `/picked` prefix is append to pickled object to distinguish them in the base hdf5 file. 

Just run `pickled_hdf5.py` for a minimal usage example.
