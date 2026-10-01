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

@cuda.jit
def gpu_process_1d(img, out):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < img.shape[0]:
        g = np.uint8((img[tidx, 0] + img[tidx, 1] + img[tidx, 2]) / 3)
        out[tidx,0], out[tidx,1], out[tidx,2] = g,g,g

if __name__ == "__main__":
    block_size = [(8,8), (8,16), (8,32), (16,16), (16,8), (16,32), (32,8), (32,16), (32,32)]
    block_size_1d = [64, 128, 256, 512, 1024]  
    pixel_count = width * height
    repeat = 20  

    gpu_time = {}
    gpu_data = cuda.to_device(np_img)
    devData = cuda.device_array(np_img.shape, dtype=np.uint8)

    # Run it first to avoid compilation time
    gpu_process[(1920//8, 1080//8), (8,8)](gpu_data, devData)

    for bls in block_size:
        grid = (int(np.ceil(width / bls[0])), int(np.ceil(height / bls[1])))

        start = time.perf_counter()
        for _ in range(repeat):
            gpu_process[grid, bls](gpu_data, devData)
        cuda.synchronize()
        end = time.perf_counter()
        gpu_time[bls] = (end - start) / repeat

    out_gpu = devData.copy_to_host()
    Image.fromarray(out_gpu).save("gpu_grey.png")

    # 1D version (lab 3) on the flattened (pixel_count, 3) image
    gpu_time_1d = {}
    flat_data = cuda.to_device(np_img.reshape(-1, 3))
    flat_out = cuda.device_array(flat_data.shape, dtype=np.uint8)
    gpu_process_1d[1, 1](flat_data, flat_out)

    for bls in block_size_1d:
        grid = int(np.ceil(pixel_count / bls))

        start = time.perf_counter()
        for _ in range(repeat):
            gpu_process_1d[grid, bls](flat_data, flat_out)
        cuda.synchronize()
        end = time.perf_counter()
        gpu_time_1d[bls] = (end - start) / repeat

    out_gpu_1d = flat_out.copy_to_host().reshape(np_img.shape)

    cpu_time = []
    for i in range(3):
        start = time.perf_counter()
        out_cpu = cpu_process(one_d_img)
        end = time.perf_counter()
        cpu_time.append(end - start)

    cpu_time = np.mean(cpu_time)
    out_cpu = out_cpu.reshape(np_img.shape)
    Image.fromarray(out_cpu).save("cpu_grey.png")

    print("2D:", gpu_time)
    print("1D:", gpu_time_1d)
    print("CPU:", cpu_time)

    c2d, c1d, ccpu = "#2a6fb0", "#8e44ad", "#d9822b"
    fig, (ax_heat, ax_cmp, ax_bar) = plt.subplots(1, 3, figsize=(19, 5.5), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})

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
    ax_heat.set_title("2D: time per block shape (ms)")
    threshold = np.nanmin(grid_ms) + (np.nanmax(grid_ms) - np.nanmin(grid_ms)) / 2
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid_ms[i, j]
            if not np.isnan(v):
                ax_heat.text(j, i, f"{v:.3f}\n({xs[j] * ys[i]} thr)", ha="center", va="center",
                             fontsize=9, color="white" if v > threshold else "#1a1a1a")
    fig.colorbar(im, ax=ax_heat, label="Execution time (ms)")

    # Middle: 1D vs 2D at the same number of threads per block
    totals = sorted(gpu_time_1d)
    pos = range(len(totals))
    t1d = [gpu_time_1d[n] * 1e3 for n in totals]
    best2d = [min((b for b in gpu_time if b[0] * b[1] == n), key=gpu_time.get, default=None) for n in totals]
    t2d = [gpu_time[b] * 1e3 if b else np.nan for b in best2d]

    ax_cmp.plot(pos, t1d, color=c1d, linewidth=2, marker="o", markersize=8, label="1D block")
    ax_cmp.plot(pos, t2d, color=c2d, linewidth=2, marker="s", markersize=8, label="2D block (best shape)")
    # all 2D shapes as faint dots, so the spread between shapes is visible too
    for b, t in gpu_time.items():
        ax_cmp.scatter(totals.index(b[0] * b[1]), t * 1e3, color=c2d, alpha=0.3, s=25, zorder=1)
    for x, b, t in zip(pos, best2d, t2d):
        if b:
            ax_cmp.annotate(f"{b[0]}x{b[1]}", (x, t), textcoords="offset points", xytext=(0, -16),
                            ha="center", fontsize=8, color="#444444")
    ax_cmp.set_xticks(list(pos), labels=totals)
    ax_cmp.set_xlabel("Threads per block")
    ax_cmp.set_ylabel("Execution time (ms)")
    ax_cmp.set_ylim(bottom=0)
    ax_cmp.set_title("1D vs 2D at equal threads per block")
    ax_cmp.grid(axis="y", linestyle="--", alpha=0.3)
    ax_cmp.set_axisbelow(True)
    ax_cmp.legend(frameon=False, loc="lower left")

    # Right: CPU vs best 1D / best 2D (log scale)
    best = min(gpu_time, key=gpu_time.get)
    best_1d = min(gpu_time_1d, key=gpu_time_1d.get)
    labels = [f"GPU 2D\n{best[0]}x{best[1]}", f"GPU 1D\n{best_1d}", "CPU"]
    times = [gpu_time[best], gpu_time_1d[best_1d], cpu_time]

    bars = ax_bar.barh(labels, times, color=[c2d, c1d, ccpu], height=0.6)
    ax_bar.set_xscale("log")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Execution time (seconds, log scale)")
    ax_bar.set_title("Best of each vs CPU")
    ax_bar.grid(axis="x", which="both", linestyle="--", alpha=0.3)
    ax_bar.set_axisbelow(True)
    for bar, elapsed in zip(bars, times):
        speedup = f"  (x{cpu_time / elapsed:,.0f})" if elapsed != cpu_time else ""
        ax_bar.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                    f"{elapsed:.2e}s{speedup}", va="center", fontsize=9)
    ax_bar.set_xlim(right=max(times) * 30)

    fig.suptitle(f"Grayscale conversion, {width}x{height} image: 1D vs 2D blocks (mean of {repeat} runs)")
    fig.tight_layout()
    fig.savefig("execution_time.png", dpi=200)