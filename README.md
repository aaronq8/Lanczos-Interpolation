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

**Implemented: naive direct 2D Lanczos-3 and a faster, linearly separable scalar Lanczos-3 resampler.**

- Loads JPEG images using `stb_image`, requesting 8-bit, three-channel RGB output.
- Splits interleaved RGB into three planar 8-bit channels.
- Resizes each color plane independently using Lanczos-3.
- Uses pixel-center destination-to-source coordinate mapping.
- Implements both a direct 2D kernel (baseline) and a two-pass, horizontal-then-vertical separable kernel.
- Uses a floating-point scratch buffer for intermediate horizontal results, avoiding intermediate 8-bit clamping and quantization.
- Normalizes filter weights at each stage, including near image borders.
- Converts final values to 8-bit RGB, interleaves the planes, and writes PNG output with `stb_image_write`.
- Produces visually correct resized output for the current test image.

**Implementation convention:** `x = rows (height)` and `y = columns (width)`. The code uses `img[row * width + col]` for row-major plane indexing.

## Performance Log

The benchmark measures the Lanczos resizing stage for **all three RGB planes**, excluding JPEG decoding and PNG writing. Measurements below are from individual reported Release runs, not statistical medians. Compare speedups **within the same machine**, not across different machines.

**Common test configuration**

| Setting | Value |
|---|---|
| Input | `sample-images/0.jpg` |
| Input resolution | 736 × 535 (width × height) |
| Output resolution | 1472 × 1070 (2× along each axis) |
| Input/output format | RGB8 (three planar channels during resampling) |
| Kernel | Lanczos-3, support radius 3 |
| Build | CMake Release; no ASan for benchmarking |
| Measured work | Resize all three RGB planes |

### Benchmark Results

| Platform | Implementation | Kernel time | Speedup vs platform baseline |
|---|---|---:|---:|
| Intel Core i9-13980HX (ROG Strix G634JZ, Ubuntu) | Naive scalar, direct 2D | **1,896.13 ms** | **1.00×** |
| Apple M1 Pro (macOS, Apple Clang) | Naive scalar, direct 2D | **~1,100 ms** | **1.00×** |
| Apple M1 Pro (macOS, Apple Clang) | Separable scalar, horizontal + vertical | **~250 ms** | **~4.40×** |
| Intel Core i9-13980HX (ROG Strix G634JZ, Ubuntu) | Separable scalar | Not yet measured | — |

On the M1 Pro, separable filtering reduced runtime by approximately **77.3%** relative to its own direct-2D baseline.

The ~1,100 ms and ~250 ms M1 Pro results are approximate figures. Compiler versions, CPU scheduling, clock behavior, and system math-library implementations can influence cross-platform timings; no cross-platform performance conclusions are drawn from these runs alone.

**Original Intel baseline log**

```text
$ ./build/release/lanczos
[INFO] Loading image from sample-images....
[INFO] Loaded image....x : 535, y : 736, channels : 3
[INFO] Lanczos Kernel elapsed : 1896.13 ms
[INFO] writing png...
```

### Linearly Separable Lanczos-3

**Observation:** A direct 2D Lanczos-3 filter combines horizontal and vertical weights for each neighboring source pixel, performing up to 36 weighted source-sample contributions per interior destination pixel. Its kernel is separable, allowing the same mathematical filtering operation to be computed using two one-dimensional passes.

**Change:**

1. **Horizontal pass:** for each of the `x` source rows, resize `y` columns to `res_y` columns, writing normalized floating-point results to a scratch buffer of shape `x × res_y`.
2. **Vertical pass:** for each output column, use neighboring scratch-buffer rows to produce the final `res_x × res_y` image. Normalize, then convert to 8-bit values and clamp to `[0, 255]`.

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

**Why this is faster:** For a 2× upscale, the direct filter performs about `36 × (2x) × (2y) = 144xy` sample contributions per channel. The separable version performs about `6 × x × (2y) + 6 × (2x) × (2y) = 36xy` contributions per channel (ignoring boundaries). It also dramatically reduces repeated Lanczos/sine evaluations. The scratch buffer adds memory traffic, but avoids much of the redundant arithmetic.

**Result:** M1 Pro runtime went from **~1,100 ms to ~250 ms** (**~4.4× faster**).

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


## Scope

- **Current:** RGB8 images, scalar CPU, Lanczos-3, 2× upscale test, direct and separable implementations.
- **Not yet implemented:** arbitrary scale ratios, 16-bit channels, alpha-aware filtering, downscaling antialiasing, SIMD, CUDA.

New measurements should be **appended**, retaining earlier platform-specific baselines and documenting the exact test configuration.
