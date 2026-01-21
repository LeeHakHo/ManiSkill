import h5py
with h5py.File('custom_demos/RaiseCube-v1/real_data/trajectory.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
    f.visititems(print_structure)


with h5py.File('demos/RaiseCube-v1/motionplanning/trajectory_300.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
    f.visititems(print_structure)