import os

def find_cuda_path():
    # Check if CUDA_HOME is set in environment variables
    cuda_home = os.environ.get('CUDA_HOME', None)
    if cuda_home:
        print(f"CUDA_HOME environment variable is set to: {cuda_home}")
    else:
        print("CUDA_HOME environment variable is not set.")

    # Common CUDA installation paths
    common_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-11',
        '/usr/local/cuda-10',
        '/usr/local/cuda-12',
        '/opt/cuda',
        '/opt/cuda-11',
        '/opt/cuda-10',
        '/opt/cuda-12'
    ]

    # Check which common path exists
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found CUDA directory at: {path}")

    # Check for nvcc in the system path
    from shutil import which
    nvcc_path = which('nvcc')
    if nvcc_path:
        print(f"Found nvcc at: {nvcc_path}")
        cuda_dir = os.path.dirname(os.path.dirname(nvcc_path))
        print(f"Assuming CUDA directory from nvcc path: {cuda_dir}")
    else:
        print("nvcc not found in PATH.")

find_cuda_path()
