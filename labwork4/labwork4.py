# Convert an RGB into grey-scale using numba CUDA.
from numba import cuda
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

rgb_img = Image.open("/mnt/c/Users/SFX16-51G/Downloads/week.jpg").resize((1920,1080))
np_img = np.array(rgb_img)
width = np_img.shape[1]
height = np_img.shape[0]

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
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x # col = 1920
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y # row = 1080

    g = np.uint8((img[y, x, 0] + img[y, x, 1] + img[y, x, 2]) / 3)
    out[y,x,0], out[y,x,1], out[y,x,2] = g,g,g

if __name__ == "__main__":
    block_size = [(8,8), (8,16), (8,32), (16,16), (16,8), (16,32), (32,8), (32,16), (32,32)]
    pixel_count = width * height

    gpu_time = {}
    gpu_data = cuda.to_device(np_img)
    devData = cuda.device_array(np_img.shape, dtype=np.uint8)

    # Run it first to avoid compilation time
    gpu_process[(1920//8, 1080//8), (8,8)](gpu_data, devData)
    
    for bls in block_size:
        grid = (int(np.ceil(width / bls[0])), int(np.ceil(height / bls[1])))

        start = time.perf_counter()
        gpu_process[grid, bls](gpu_data, devData)
        cuda.synchronize()  
        end = time.perf_counter()
        gpu_time[bls] = end - start

    out_gpu = devData.copy_to_host()
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

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: heatmap of GPU time (ms) over blockDim.x (columns) x blockDim.y (rows)
    xs = sorted({b[0] for b in gpu_time})
    ys = sorted({b[1] for b in gpu_time})
    grid_ms = np.full((len(ys), len(xs)), np.nan)
    for (bx, by), t in gpu_time.items():
        grid_ms[ys.index(by), xs.index(bx)] = t * 1e3

    im = ax_heat.imshow(grid_ms, cmap="Blues", origin="lower")
    ax_heat.set_xticks(range(len(xs)), labels=xs)
    ax_heat.set_yticks(range(len(ys)), labels=ys)
    ax_heat.set_xlabel("blockDim.x (threads)")
    ax_heat.set_ylabel("blockDim.y (threads)")
    ax_heat.set_title("GPU time per 2D block shape (ms)")
    threshold = np.nanmin(grid_ms) + (np.nanmax(grid_ms) - np.nanmin(grid_ms)) / 2
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid_ms[i, j]
            if not np.isnan(v):
                ax_heat.text(j, i, f"{v:.3f}\n({xs[j] * ys[i]} thr)", ha="center", va="center",
                             fontsize=9, color="white" if v > threshold else "#1a1a1a")
    fig.colorbar(im, ax=ax_heat, label="Execution time (ms)")

    # Right: CPU vs fastest / slowest GPU block (log scale)
    best = min(gpu_time, key=gpu_time.get)
    worst = max(gpu_time, key=gpu_time.get)
    labels = [f"GPU fastest\n{best[0]}x{best[1]}", f"GPU slowest\n{worst[0]}x{worst[1]}", "CPU"]
    times = [gpu_time[best], gpu_time[worst], cpu_time]

    bars = ax_bar.barh(labels, times, color=["#2a6fb0", "#8fb4d9", "#d9822b"], height=0.6)
    ax_bar.set_xscale("log")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Execution time (seconds, log scale)")
    ax_bar.set_title(f"CPU vs GPU - speedup x{cpu_time / gpu_time[best]:,.0f} (fastest block)")
    ax_bar.grid(axis="x", which="both", linestyle="--", alpha=0.3)
    ax_bar.set_axisbelow(True)
    for bar, elapsed in zip(bars, times):
        ax_bar.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                    f"{elapsed:.2e}s", va="center", fontsize=9)
    ax_bar.set_xlim(right=max(times) * 10)

    fig.suptitle(f"Grayscale conversion, {width}x{height} image, 2D blocks")
    fig.tight_layout()
    fig.savefig("execution_time.png", dpi=200)
    plt.show()