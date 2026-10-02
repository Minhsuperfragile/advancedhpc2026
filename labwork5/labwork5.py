from numba import cuda, njit, prange, get_num_threads
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

rgb_img = Image.open("/mnt/c/Users/SFX16-51G/Downloads/random_img.jpg").resize((512,512))
np_img = np.array(rgb_img).astype(np.float32)
width = np_img.shape[1]
height = np_img.shape[0]

    
@cuda.jit
def gpu_process(img, out):
    x, y = cuda.grid(2)
    g = (img[y, x, 0] + img[y, x, 1] + img[y, x, 2]) / 3
    out[y,x,0], out[y,x,1], out[y,x,2] = g,g,g

@cuda.jit
def perform_gaussian_blur(img, k, out):
    x,y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y # index for the shared memory

    # shared memory
    sm = cuda.shared.array(shape=(8, 14), dtype=np.float32)
    cx = min(max(x,0), img.shape[1] - 1)
    cy = min(max(y,0), img.shape[0] - 1)
    sm[ty,tx + 3] = img[cy, cx, 0]
    if tx < 3:                               
        sm[ty, tx] = img[cy, min(max(cx-3, 0), img.shape[1] - 1), 0]
    if tx >= 8 - 3:                          
        sm[ty, tx+6] = img[cy, min(max(cx+3, 0), img.shape[1] - 1), 0]
    cuda.syncthreads()

    if x < img.shape[1] and y < img.shape[0]:
        pixel = 0.
        for i in range(7):
            pixel += k[i] * sm[ty, tx + i]
        out[y,x,0], out[y,x,1], out[y,x,2] = pixel,pixel,pixel

@cuda.jit
def perform_gaussian_blur2(img,k,out):
    x,y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    sm = cuda.shared.array(shape=(14,8), dtype=np.float32)
    cx = min(max(x,0), img.shape[1] - 1)
    cy = min(max(y,0), img.shape[0] - 1)
    sm[ty+3, tx] = img[cy,cx,0]
    if ty < 3:
        sm[ty,tx] = img[min(max(cy-3,0), img.shape[0] - 1), cx, 0]
    if ty >= 8-3:
        sm[ty+6, tx] = img[min(max(cy+3,0), img.shape[0] - 1), cx, 0]
    cuda.syncthreads()

    if x < img.shape[1] and y < img.shape[0]:
        pixel = 0. 
        for i in range(7):
            pixel += k[i] * sm[ty + i, tx]
        out[y,x,0], out[y,x,1], out[y,x,2] = pixel,pixel,pixel

@cuda.jit
def perform_gaussion_blur3(img, k, out): # no shared mem
    x, y = cuda.grid(2)
    if x >= img.shape[1] or y >= img.shape[0]: return

    pixel = 0
    for i in range(7):
        col = min(max(x - 3 + i, 0), img.shape[1] - 1)
        pixel += k[i] * img[y, col, 0]
    out[y, x, 0], out[y, x, 1], out[y, x, 2] = pixel, pixel, pixel

@cuda.jit
def perform_gaussian_blur4(img, k, out):
    x, y = cuda.grid(2)
    if x >= img.shape[1] or y >= img.shape[0]: return

    pixel = 0
    for i in range(7):
        row = min(max(y - 3 + i, 0), img.shape[0] - 1)
        pixel += k[i] * img[row, x, 0]
    out[y, x, 0], out[y, x, 1], out[y, x, 2] = pixel, pixel, pixel

# CPU versions: 
def _cpu_grey(img, out):
    for y in prange(img.shape[0]):
        for x in range(img.shape[1]):
            g = (img[y, x, 0] + img[y, x, 1] + img[y, x, 2]) / 3
            out[y, x, 0], out[y, x, 1], out[y, x, 2] = g, g, g

def _cpu_blur_row(img, k, out): # same as perform_gaussion_blur3
    for y in prange(img.shape[0]):
        for x in range(img.shape[1]):
            pixel = 0.
            for i in range(7):
                col = min(max(x - 3 + i, 0), img.shape[1] - 1)
                pixel += k[i] * img[y, col, 0]
            out[y, x, 0], out[y, x, 1], out[y, x, 2] = pixel, pixel, pixel

def _cpu_blur_col(img, k, out): # same as perform_gaussian_blur4
    for y in prange(img.shape[0]):
        for x in range(img.shape[1]):
            pixel = 0.
            for i in range(7):
                row = min(max(y - 3 + i, 0), img.shape[0] - 1)
                pixel += k[i] * img[row, x, 0]
            out[y, x, 0], out[y, x, 1], out[y, x, 2] = pixel, pixel, pixel

cpu_grey_single, cpu_grey_multi = njit(_cpu_grey), njit(parallel=True)(_cpu_grey)
cpu_blur_row_single, cpu_blur_row_multi = njit(_cpu_blur_row), njit(parallel=True)(_cpu_blur_row)
cpu_blur_col_single, cpu_blur_col_multi = njit(_cpu_blur_col), njit(parallel=True)(_cpu_blur_col)

def bench(fn, repeat, sync=False):
    fn() # warm up
    if sync: cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if sync: cuda.synchronize() 
    return (time.perf_counter() - start) / repeat

if __name__ == "__main__":
    time_result = {}
    repeat = 20
    block_size = (8,8) # 64 threads,  warps
    grid = (512//8, 512//8)

    pixel_count = width * height
    gpu_data = cuda.to_device(np_img)
    devData = cuda.device_array(np_img.shape, dtype=np.float32)
    gpu_process[grid, block_size](gpu_data, devData)

    # Gauss convolution is separable, we use it to reduce the number of operation.
    # Cite: https://medium.com/@RaymondTayBL/gaussian-blur-separable-convolution-vs-full-2d-convolution-d5e9b64bf84f
    # We perform conv on row 1D kernel, then take the result to perform the next conv on col 1D kernel.
    gauss_kernel_row = np.array([1,6,15,20,15,6,1]).astype(np.float32)
    gauss_kernel_row = gauss_kernel_row / np.sum(gauss_kernel_row)
    gauss_kernel_col = np.transpose(gauss_kernel_row)

    dev_k = cuda.to_device(gauss_kernel_row)
    blurred_dev = cuda.device_array_like(np_img)
    blurred_out = cuda.device_array_like(np_img)

    def gpu_shared():
        perform_gaussian_blur[grid, block_size](devData, dev_k, blurred_dev)
        perform_gaussian_blur2[grid, block_size](blurred_dev, dev_k, blurred_out)

    def gpu_global():
        perform_gaussion_blur3[grid, block_size](devData, dev_k, blurred_dev)
        perform_gaussian_blur4[grid, block_size](blurred_dev, dev_k, blurred_out)

    time_result["gpu_shared_mem"] = bench(gpu_shared, repeat, sync=True)
    gpu_out_shared = blurred_out.copy_to_host()
    time_result["gpu_global_mem"] = bench(gpu_global, repeat, sync=True)
    gpu_out_global = blurred_out.copy_to_host()

    # CPU: same pipeline, grey first (not timed, like the GPU), then row blur + col blur
    cpu_grey = np.empty_like(np_img)
    cpu_tmp = np.empty_like(np_img)
    cpu_out = np.empty_like(np_img)
    cpu_grey_single(np_img, cpu_grey)

    time_result["cpu_single_core"] = bench(lambda: (cpu_blur_row_single(cpu_grey, gauss_kernel_row, cpu_tmp),
                                                    cpu_blur_col_single(cpu_tmp, gauss_kernel_row, cpu_out)), repeat)
    cpu_out_single = cpu_out.copy()
    time_result["cpu_multi_thread"] = bench(lambda: (cpu_blur_row_multi(cpu_grey, gauss_kernel_row, cpu_tmp),
                                                     cpu_blur_col_multi(cpu_tmp, gauss_kernel_row, cpu_out)), repeat)
    cpu_out_multi = cpu_out.copy()

    # all four versions must produce the same image
    for name, res in [("gpu_global_mem", gpu_out_global), ("cpu_single_core", cpu_out_single), ("cpu_multi_thread", cpu_out_multi)]:
        print(f"{name} matches gpu_shared_mem: {np.allclose(res, gpu_out_shared, atol=1e-3)}")

    for name, t in time_result.items():
        print(f"{name:>17}: {t * 1e3:.4f} ms")

    Image.fromarray(gpu_out_shared.astype(np.uint8)).save("blurred_grey.png")
    Image.fromarray(cpu_out_multi.astype(np.uint8)).save("blurred_grey_cpu.png")

    # Comparison plot: CPU (orange) vs GPU (blue), speedup relative to CPU single core
    c_cpu, c_gpu = "#eb6834", "#2a78d6"
    entries = [("CPU single core", time_result["cpu_single_core"], c_cpu),
               (f"CPU multi-thread\n({get_num_threads()} threads)", time_result["cpu_multi_thread"], c_cpu),
               ("GPU global mem", time_result["gpu_global_mem"], c_gpu),
               ("GPU shared mem", time_result["gpu_shared_mem"], c_gpu)]
    labels = [e[0] for e in entries]
    times_ms = [e[1] * 1e3 for e in entries]
    base = times_ms[0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, times_ms, color=[e[2] for e in entries], height=0.6, edgecolor="white", linewidth=2)
    ax.invert_yaxis()
    ax.set_xlabel("Execution time (ms)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for bar, t in zip(bars, times_ms):
        speedup = f"  (x{base / t:,.1f})" if t != base else ""
        ax.text(bar.get_width() + base * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f} ms{speedup}", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, max(times_ms) * 1.3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c_cpu), plt.Rectangle((0, 0), 1, 1, color=c_gpu)],
              labels=["CPU", "GPU"], frameon=False, loc="lower right")
    ax.set_title(f"Separable 7-tap Gaussian blur, {width}x{height} image (mean of {repeat} runs)")
    fig.tight_layout()
    fig.savefig("execution_time.png", dpi=200)