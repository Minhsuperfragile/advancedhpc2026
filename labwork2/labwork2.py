from numba import cuda


def to_gib(number_of_bytes):
    return number_of_bytes / (1024**3)

device = cuda.current_context().device
memory = cuda.current_context().get_memory_info()

cores_per_sm = 128
core_count = (
    device.MULTIPROCESSOR_COUNT * cores_per_sm
    if cores_per_sm is not None
    else "Unknown"
)

print(f"Device name: {device.name}")
print(f"Id: {device.id}")
print(f"Multiprocessor count: {device.MULTIPROCESSOR_COUNT}")
print(f"Core count: {core_count}")
print(f"Memory size: {to_gib(memory.total):.1f} Gb")
print(f"Memory clock: {device.MEMORY_CLOCK_RATE / 1000:.0f} MHz")
print(f"GPU clock: {device.CLOCK_RATE / 1000:.0f} MHz")
