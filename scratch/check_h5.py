import h5py
with h5py.File('/home/jyuan/.stable-wm/pusht_expert_train.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    for k in f.keys():
        try:
            print(k, f[k].shape, f[k].dtype)
        except Exception as e:
            print(k, "Error:", e)
