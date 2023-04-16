import h5py

filename = "E:/our_model.h5"

h5 = h5py.File(filename , 'r')
keys=h5.items()
for i in keys:
    print(i)

# futures_data = h5['futures_data']  # VSTOXX futures data
# options_data = h5['options_data']  # VSTOXX call option data

h5.close()