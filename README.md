# Lanczos Image Resampling

README is AI Generated, Code is handwritten.

Goal: implement and optimize **Lanczos-3 image resampling**, beginning with a straightforward scalar CPU implementation and progressing toward CPU SIMD and CUDA versions.

The emphasis is on understanding interpolation, numerical behavior, memory layout, profiling, and performance optimization—not on using a prebuilt resize implementation.

## Visual Comparison

### Original Image (736 × 535)

![Original Image](sample-images/0.jpg)

### Lanczos-3 Resized (1472 × 1070)

![Lanczos-3 Resized](sample-images/resized-sep.png)

## Current Status

**Implemented: naive direct 2D Lanczos-3, separable scalar Lanczos-3, and per-coordinate precomputed Lanczos weights.**

- Loads JPEG images using `stb_image`, requesting 8-bit, three-channel RGB output.
- Splits interleaved RGB into three planar 8-bit channels.
- Resizes each color plane independently using Lanczos-3.
- Uses pixel-center destination-to-source coordinate mapping.
- Implements both a direct 2D kernel (baseline) and a two-pass, horizontal-then-vertical separable kernel.
- Precomputes Lanczos-3 coefficients for each destination column in the horizontal pass and destination row in the vertical pass, reusing them across the opposite axis.
- Uses a floating-point scratch buffer for intermediate horizontal results, avoiding intermediate 8-bit clamping and quantization.
- Normalizes filter weights at each stage, including near image borders.
- Converts final values to 8-bit RGB, interleaves the planes, and writes PNG output with `stb_image_write`.
- Produces visually correct resized output for the current test image.

**Implementation convention:** `x = rows (height)` and `y = columns (width)`. The code uses `img[row * width + col]` for row-major plane indexing.

### Optimized Scalar Baseline — Intel i9-13980HX

| Parameter | Value |
|---|---|
| Input resolution | 736 × 535 |
| Output resolution | 1472 × 1070 |
| Color format | RGB8, planar |
| Algorithm | Separable Lanczos-3 |
| Optimization | Precomputed weights per destination coordinate |
| SIMD | None (explicit SIMD intrinsics) |
| Multithreading | None |
| Benchmark | Google Benchmark |
| Repetitions | 10 |
| Median latency | **21.655 ms** |
| Standard deviation | 0.111 ms |
| Coefficient of variation | 0.51% |


### Benchmark Results

| Platform | Implementation / change | Kernel time | Speedup vs same-platform baseline |
|---|---|---:|---:|
| Intel Core i9-13980HX (Ubuntu) | Naive scalar, direct 2D | **1,896.13 ms** | 1.00× |
| Intel Core i9-13980HX (Ubuntu) | Separable + precomputed weights in both passes | **22.90 ms** | **82.82×** |
| Apple M1 Pro (macOS, Apple Clang) | Naive scalar, direct 2D | **~1,100 ms** | 1.00× |
| Apple M1 Pro (macOS, Apple Clang) | Separable scalar (initial) | **~250 ms** | ~4.40× |
| Apple M1 Pro (macOS, Apple Clang) | Separable + vertical-row traversal | **244.43 ms** | ~4.50× |
| Apple M1 Pro (macOS, Apple Clang) | Separable + vertical weight precomputation | **98.10 ms** | ~11.21× |

**Intel latest: ~98.8% less time** than the original naive 2D baseline. The **22.90 ms** figure is a single reported run, not a median. M1 Pro timings also come from individual runs, with the ~1,100 ms and ~250 ms values approximate. The **Intel and M1 Pro timings are separate experiments**; do not calculate optimization speedups across machines.

**Intel latest per-channel breakdown (ms)**

| Channel | Horizontal | Vertical |
|---|---:|---:|
| R | 5.8977 | 5.4353 |
| G | 2.2811 | 3.6287 |
| B | 2.1836 | 3.2636 |
| **Sum** | **10.3624** | **12.3276** |

The reported total kernel time is **22.8961 ms** (including small overhead outside pass timers). The first channel took longer than subsequent channels; this is not yet explained, and the benchmark should be repeated under controlled conditions.

**Original Intel baseline log**

```text
$ ./build/release/lanczos
[INFO] Loading image from sample-images....
[INFO] Loaded image....x : 535, y : 736, channels : 3
[INFO] Lanczos Kernel elapsed : 1896.13 ms
[INFO] writing png...
```

**Latest Intel log (2026-10-20)**

```text
[INFO] Loaded image....x : 535, y : 736, channels : 3
[INFO] horizontal pass : 5.8977 ms
[INFO] vertical pass : 5.43532 ms
[INFO] horizontal pass : 2.28108 ms
[INFO] vertical pass : 3.62871 ms
[INFO] horizontal pass : 2.1836 ms
[INFO] vertical pass : 3.26362 ms
[INFO] Lanczos Kernel elapsed : 22.8961 ms
```

### Optimization 1: Linearly Separable Lanczos-3

**Observation:** The direct 2D Lanczos-3 kernel combines horizontal and vertical weights, requiring up to 36 source-sample contributions per interior output pixel. It factors into two one-dimensional passes.

**Change:**

1. **Horizontal pass:** resize each original row from `y` to `res_y` columns. Store the result in a floating-point scratch buffer (`x × res_y`).
2. **Vertical pass:** resize the scratch buffer from `x` to `res_x` rows; normalize and convert to `uint8_t` only at the end.

```text
Source RGB plane            x × y          (uint8_t)
       |
       | Horizontal Lanczos-3
       v
Scratch buffer              x × res_y      (float)
       |
       | Vertical Lanczos-3
       v
Resized RGB plane           res_x × res_y  (uint8_t)
```

For 2× upscaling, the direct filter needs approximately `144xy` source-sample contributions per channel, compared with `36xy` for two-pass filtering (ignoring edges). The scratch buffer adds memory traffic but avoids much of the redundant arithmetic.

**Observed on M1 Pro:** ~1,100 ms → ~250 ms (**~4.4× speedup**).

### Optimization 2: Vertical loop order and coefficient reuse (Apple M1 Pro)

**Observation:** With output columns as the outer loop, consecutive destination writes are separated by the destination row stride. Iterating destination rows outermost makes writes contiguous; for a fixed destination row, the vertical Lanczos weights are identical across all columns.

**Changes and results:**

- Reverse vertical loop order (`rx` outer, `ry` inner): **~250 ms → 244.43 ms** total (the ~250 ms initial value is approximate).
- Compute the handful of vertical `L3()` weights and their normalization sum once per output row, then reuse across destination columns: **244.43 ms → 98.10 ms** total on the M1 Pro.

This removes a large number of repeated trigonometric evaluations. Remaining sample accumulation still occurs for every output pixel.

### Optimization 3: Horizontal coefficient reuse (Intel i9-13980HX)

**Observation:** Horizontal weights depend on the destination column, not the source row. The initial horizontal pass recomputed them for every row and color plane.

**Change:** Traverse destination columns outermost. For each output column, precompute its valid neighbor range, `L3()` coefficients, and sum of weights **once**, then reuse these coefficients for all source rows. The vertical pass also precomputes weights per destination row.

**Result:** The latest Intel implementation measured **22.8961 ms** for all three RGB planes, compared with its **1,896.13 ms direct-2D baseline** (**~82.8× speedup overall**). This compares the combined optimizations, **not** the isolated benefit of horizontal coefficient precomputation; no directly comparable intermediate Intel benchmark was recorded.

**Trade-off:** Column-major traversal in the horizontal pass reuses coefficients but causes strided input/scratch accesses across successive rows. A future experiment can precompute a compact coefficient table and restore row-major traversal to test memory locality separately.

## Resampling Mathematics

### 1. Destination-to-source coordinate mapping

Map destination pixel centers back into source coordinates:

$$
x_s = (x_d + 0.5)\frac{W_s}{W_d} - 0.5
$$

$$
y_s = (y_d + 0.5)\frac{H_s}{H_d} - 0.5
$$

Here `W` is width and `H` is height; in the implementation's variable convention, columns are `y` and rows are `x`.

### 2. Lanczos-3 kernel

The normalized sinc function is:

$$
\mathrm{sinc}(t) =
\begin{cases}
1, & t = 0 \\
\frac{\sin(\pi t)}{\pi t}, & t \neq 0
\end{cases}
$$

The Lanczos-3 interpolation kernel is:

$$
L_3(t) =
\begin{cases}
\mathrm{sinc}(t)\mathrm{sinc}(t/3), & |t| < 3 \\
0, & |t| \geq 3
\end{cases}
$$

### 3. Direct two-dimensional Lanczos interpolation

Using source **row** `i` and **column** `j`, the combined weight is:

$$
w(i,j) = L_3(r_s-i)L_3(c_s-j)
$$

where `r_s` and `c_s` are mapped source row and column coordinates. The output is:

$$
D(r_d,c_d) =
\frac{\sum_{i,j} S(i,j)w(i,j)}{\sum_{i,j} w(i,j)}
$$

Only valid source pixels with $|r_s-i| < 3$ and $|c_s-j| < 3$ contribute. An interior destination pixel typically considers **6 × 6 = 36** source pixels.

### 4. Separable horizontal and vertical interpolation

Because the combined weight is a product of independent one-dimensional weights, filtering can be done in two passes.

First, horizontal filtering along columns:

$$
T(i,c_d) =
\frac{\sum_j S(i,j)L_3(c_s-j)}{\sum_j L_3(c_s-j)}
$$

Then, vertical filtering along rows:

$$
D(r_d,c_d) =
\frac{\sum_i T(i,c_d)L_3(r_s-i)}{\sum_i L_3(r_s-i)}
$$

The normalization factors multiply to the direct 2D normalization factor because:

$$
\sum_{i,j} w_x(i)w_y(j)
= \left(\sum_i w_x(i)\right)\left(\sum_j w_y(j)\right)
$$

Thus, with consistent boundary handling and without intermediate quantization, **the two-pass method is mathematically equivalent to direct 2D filtering**. Small floating-point differences are possible in practice.

### 5. Output conversion

The scratch buffer retains floating-point values without clamping. Only after the vertical pass is the final value converted to 8 bits:

$$
D_{\mathrm{8bit}} = \mathrm{clamp}(\mathrm{trunc}(D), 0, 255)
$$

The formula uses **truncation** to match the current C++ cast to `int`, not rounding to nearest.

## Building

The project uses CMake and local `stb_image` / `stb_image_write` headers.

```bash
# Debug build (with ASan if ENABLE_ASAN is active in Debug)
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j

# Optimized Release build, ASan disabled
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=OFF
cmake --build build/release -j

# Run from the repository root
./build/release/lanczos
```

Image paths are currently hard-coded in the executable; update the stated output filename as the program evolves. Historical Release logs include `[INFO]` messages; verify logger configuration if builds are intended to suppress all logs.

## Next Optimizations

- [x] Naive direct 2D Lanczos-3 implementation
- [x] Horizontal + vertical separable scalar filtering with float scratch buffer
- [ ] Repeat each benchmark and report median / spread, keeping CPU and compiler settings documented
- [ ] Quantitatively compare separable output against the original direct-2D reference (MAE, maximum absolute error, PSNR)
- [x] Precompute and reuse horizontal/vertical weights across rows/columns
- [ ] Evaluate shared coefficient tables across RGB planes
- [x] Experiment with vertical row-major traversal
- [ ] Evaluate horizontal row-major traversal with precomputed coefficient tables and scratch-buffer locality
- [ ] Profile using Linux `perf` / platform-appropriate profiling tools
- [ ] SIMD implementation (AVX2 on x86-64; ARM NEON as an optional Apple Silicon experiment)
- [ ] CUDA implementation and CPU–GPU comparisons

## Scope

- **Current:** RGB8 images, scalar CPU, Lanczos-3, 2× upscale test, direct and separable implementations.
- **Not yet implemented:** arbitrary scale ratios, 16-bit channels, alpha-aware filtering, downscaling antialiasing, SIMD, CUDA.

New measurements should be **appended**, retaining earlier platform-specific baselines and documenting the exact test configuration.
