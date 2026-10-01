# Convert an RGB into grey-scale using numba CUDA.
from numba import cuda
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
import time

rgb_img = imread(fname="/mnt/c/Users/SFX16-51G/Downloads/week.jpg")
np_img = np.array(rgb_img)
one_d_img = np_img.flatten()

def cpu_process(img: np.ndarray):
    out = np.zeros_like(img)
    img = img.astype(np.uint16)
    for pixels in range (0,img.size,3):
        g = np.uint8((img[pixels] + img[pixels+1] + img[pixels+2])/3)
        out[pixels], out[pixels+1], out[pixels+2] = g,g,g
    return out
    
@cuda.jit
def gpu_process(img, out):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((img[tidx, 0] + img[tidx, 1] + img[tidx, 2]) / 3)
    out[tidx,0], out[tidx,1], out[tidx,2] = g,g,g

if __name__ == "__main__":
    block_size = [32,64,128,256]
    pixel_count = np_img.shape[0] * np_img.shape[1]

    gpu_time = {}
    gpu_data = cuda.to_device(np_img.reshape(-1,3))
    devData = cuda.device_array(np_img.reshape(-1,3).shape, dtype=np.uint8)

    # Run it first to avoid compilation time
    gpu_process[1,1](gpu_data, devData)
    
    for bls in block_size:
        grid = int(np.ceil(pixel_count / bls))

        start = time.perf_counter()
        gpu_process[grid, bls](gpu_data, devData)
        cuda.synchronize()
        end = time.perf_counter()
        gpu_time[bls] = end - start

    out_gpu = devData.copy_to_host().reshape(np_img.shape)
    Image.fromarray(out_gpu).save("gpu_grey.png")

    cpu_time = []
    for i in range(3):
        start = time.perf_counter()
        out_cpu = cpu_process(one_d_img)
        end = time.perf_counter()
        cpu_time.append(end - start)

    cpu_time = np.mean(cpu_time)
    out_cpu = out_cpu.reshape(np_img.shape)
    Image.fromarray(out_cpu).save("cpu_grey.png")

    print(out_cpu.shape, out_gpu.shape)
    print(gpu_time) 
    print(cpu_time)

    import matplotlib.pyplot as plt
    labels = [f"GPU\n{block} threads" for block in gpu_time]
    times = list(gpu_time.values())

    labels.append("CPU")
    times.append(cpu_time)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, times, color=["steelblue"] * len(gpu_time) + ["darkorange"])

    plt.yscale("log")
    plt.ylabel("Execution time (seconds, log scale)")
    plt.xlabel("Processing configuration")
    plt.title("CPU vs GPU grayscale processing time")
    plt.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    for bar, elapsed in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{elapsed:.2e}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("execution_time.png", dpi=200)
    plt.show()