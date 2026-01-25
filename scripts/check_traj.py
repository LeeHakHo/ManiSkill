import h5py

print("============EXPERT DATASET============")

with h5py.File('custom_demos/RaiseCube-v1/real_data/trajectory.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
    f.visititems(print_structure)

print("============MANISKILL DATASET============")

with h5py.File('demos/RaiseCube-v1/motionplanning/trajectory_100.rgb.pd_ee_delta_pose.physx_cpu.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
    f.visititems(print_structure)