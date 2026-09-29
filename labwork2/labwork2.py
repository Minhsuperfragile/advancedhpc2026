from numba import cuda


def to_gib(number_of_bytes):
    return number_of_bytes / (1024**3)

context = cuda.current_context()
device = context.device
memory = context.get_memory_info()

major = device.compute_capability.major
minor = device.compute_capability.minor
compute_capability = (major, minor)

cores_per_sm_table = {
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere data-center GPUs
    (8, 6): 128,  # Ampere consumer GPUs
    (8, 7): 128,  # Ampere embedded GPUs
    (8, 9): 128,  # Ada Lovelace
}

cores_per_sm = cores_per_sm_table.get(compute_capability)
sm_count = device.MULTIPROCESSOR_COUNT

print(f"Device name:        {device.name}")
print(f"Compute capability: {major}.{minor}")
print(f"Device number:      {device.id}")

print(f"Streaming multiprocessors:       {sm_count}")
print(f"Warp size:                       {device.WARP_SIZE}")
print(f"Maximum threads per block:       {device.MAX_THREADS_PER_BLOCK}")
print(
    f"Maximum threads per SM:          "
    f"{device.MAX_THREADS_PER_MULTIPROCESSOR}"
)

if cores_per_sm is not None:
    estimated_cores = sm_count * cores_per_sm
    print(f"FP32 CUDA cores per SM:           {cores_per_sm}")
    print(f"Estimated total CUDA cores:       {estimated_cores}")
else:
    print("Estimated total CUDA cores:       Unknown architecture")

print(f"Total GPU memory:     {to_gib(memory.total):.2f} GiB")
print(f"Free GPU memory:      {to_gib(memory.free):.2f} GiB")
print(f"Used GPU memory:      {to_gib(memory.total - memory.free):.2f} GiB")
print(
    f"L2 cache:             "
    f"{device.L2_CACHE_SIZE / (1024**2):.2f} MiB"
)
print(
    f"Shared memory/block:  "
    f"{device.MAX_SHARED_MEMORY_PER_BLOCK / 1024:.0f} KiB"
)
print(f"Registers/block:      {device.MAX_REGISTERS_PER_BLOCK}")

print(f"GPU clock:            {device.CLOCK_RATE / 1000:.0f} MHz")
print(f"Memory clock:         {device.MEMORY_CLOCK_RATE / 1000:.0f} MHz")

print(
    "Maximum block dimensions: "
    f"{device.MAX_BLOCK_DIM_X} × "
    f"{device.MAX_BLOCK_DIM_Y} × "
    f"{device.MAX_BLOCK_DIM_Z}"
)
print(
    "Maximum grid dimensions:  "
    f"{device.MAX_GRID_DIM_X} × "
    f"{device.MAX_GRID_DIM_Y} × "
    f"{device.MAX_GRID_DIM_Z}"
)