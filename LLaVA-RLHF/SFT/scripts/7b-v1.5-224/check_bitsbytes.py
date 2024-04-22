import sys
import os

# Add the CUDA library path
cuda_lib_path = '/usr/local/cuda/lib64'
os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Check if Python can see the modified environment variable
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

# Try importing bitsandbytes
import bitsandbytes as bnb

# If no errors occur, the setup might now be correct
print("Bits and Bytes loaded successfully!")
