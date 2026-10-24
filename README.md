# Lanczos Image Resampling

README is AI Generated, Code is handwritten.

Goal: implement and optimize **Lanczos-3 image resampling**, starting with a straightforward scalar CPU implementation and progressing toward CPU SIMD (AVX2 / possibly NEON) and CUDA. The emphasis is on interpolation mathematics, memory layout, profiling, and measurable performance improvements—not on calling a prebuilt resize library.

## Visual Comparison

### Original Image (736 × 535)

![Original Image](sample-images/0.jpg)

### Lanczos-3 Resized (1472 × 1070)

![Lanczos-3 Resized](sample-images/resized-sep.png)

## Current Status

**Scalar CPU implementation complete. AVX2 horizontal and vertical passes implemented and benchmarked.**

- Implemented separable Lanczos-3 interpolation using a floating-point intermediate buffer.
- Precomputed and normalized horizontal and vertical interpolation coefficients, shared across RGB channels.
- Implemented AVX2 horizontal interpolation using 8-lane multiplication and horizontal reduction.
- Implemented AVX2/FMA vertical interpolation, processing eight adjacent output pixels simultaneously.
- Added zero-padded coefficient tables with eight entries per coordinate to support AVX2 loads.
- Moved scratch-buffer and output-buffer allocations outside the resize hot path, allowing buffer reuse across RGB channels.
- Precomputed source-neighbor bounds to reduce calculations during resizing.
- Benchmarked fixed eight-iteration and variable-bound vertical loops.
- Achieved a **4.991 ms median RGB resize time** on an Intel Core i9-13980HX using Google Benchmark.

**Current best:** AVX2 horizontal + AVX2/FMA vertical with variable-bound vertical interpolation.

## Reference Scalar Baseline — Intel Core i9-13980HX

| Parameter | Value |
|---|---|
| Input | 736 × 535 RGB8 (three separate planes) |
| Output | 1472 × 1070 RGB8 |
| Algorithm | Separable Lanczos-3, 2× upscale |
| Coefficients | Precomputed and normalized per destination coordinate; shared across RGB planes |
| Horizontal / vertical traversal | Column-major / row-major |
| Explicit SIMD | None |
| Threads | Single-threaded |
| Tool | Google Benchmark; 10 repetitions |
| Controlled CPU affinity | `taskset -c 8` (Intel P-core logical CPU) |
| **RGB median, pinned run 1** | **16.501 ms** (CV 0.41%) |
| **RGB median, pinned run 2** | **16.773 ms** (wall-time CV 0.96%; CPU-time CV 0.27%) |
| Application RGB kernel, pinned, after one warm-up | ~15.33–15.63 ms across six runs |

The **~16.5–16.8 ms Google Benchmark measurements** are the current **resize-only, steady-state** scalar reference for this configuration. The benchmark calls `lanczos_init()` outside the timed `for (auto \_ : state)` loop, so coefficient-table generation is **excluded**. Allocating/freeing scratch and result buffers **inside** `lanczos_resize()` remains included. The benchmark warm-up output was held allocated until cleanup in the recorded code; preserve or standardize that behavior when comparing future implementations.

The application stopwatch and Google Benchmark have different measurement scopes and conditions; their numbers should not be treated as interchangeable. For meaningful scalar-vs-AVX2 comparisons, run both variants on the same P-core with the same inputs, build flags, allocation strategy, and warm-up protocol. The application measurements exclude image loading and PNG writing.

### Benchmark commands

```bash

*# Optimized build with debug symbols for perf; no ASan*

cmake -S . -B build/profile \\

    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\

    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fno-omit-frame-pointer" \\

    -DENABLE_ASAN=OFF

cmake --build build/profile -j

*# Pin the RGB benchmark to logical CPU 8 (a P-core on the i9-13980HX)*

taskset -c 8 ./build/profile/lanczos-bench \\

    --benchmark_filter='^BM_lanczosRGB$' \\

    --benchmark_repetitions=10 \\

    --benchmark_min_time=0.5s \\

    --benchmark_report_aggregates_only=true

*# Application measurement (the application performs a warm-up resize)*

taskset -c 8 ./build/release/lanczos-main

```

The final pinned benchmark results above were obtained with `build/release/lanczos-bench`; the `build/profile` command is provided for an optimized build that also supports source-level profiling. **Use the same actual build for both implementations** when reporting SIMD speedups.

### Latest controlled Google Benchmark results

All times below are median per RGB iteration unless otherwise noted. Input: **736 × 535**; output: **1472 × 1070**.

| Configuration | Median wall time | Mean wall time | CV (wall) | Interpretation |
|---|---:|---:|---:|---|
| Earlier scalar, in-function coefficient calculation | 21.655 ms | 21.711 ms | 0.51% | Historical Google Benchmark; different implementation/measurement scope |
| Shared precomputed normalized coefficients; `-O3` profiling build | 16.250 ms | 16.290 ms | 0.74% | An earlier, faster run; not a stable cross-session reference |
| Latest scalar, unpinned | 20.377 ms | 20.377 ms | 0.54% | Hybrid-core scheduling not controlled |
| **Latest scalar, pinned CPU 8 — run 1** | **16.501 ms** | **16.499 ms** | **0.41%** | Controlled baseline |
| **Latest scalar, pinned CPU 8 — run 2** | **16.773 ms** | **16.823 ms** | **0.96%** | Controlled baseline |

**Important:** the original ~21.655 ms benchmark computed weights during resize; the newer benchmark initializes shared coefficients outside the measured loop. Their ratio is useful historically but **not a strictly end-to-end comparison**. CPU affinity substantially reduced the observed gap between the application and benchmark timings. The pinning experiment and the warm-up experiment were not isolated in a complete factorial comparison, so the precise contribution of each remains unmeasured.

A separate **single-plane** benchmark for **1920 × 1080 input → 3840 × 2160 output** previously reported a median of **35.564 ms**, CV **0.98%**. It is **not** an RGB benchmark and was collected under an earlier configuration; do not compare it directly with the latest three-plane results.

## AVX2 Performance — October 25, 2026

### Benchmark Configuration

| Parameter | Value |
|---|---|
| CPU | Intel Core i9-13980HX |
| OS | Ubuntu Linux |
| Input | 736 × 535 RGB8 |
| Output | 1472 × 1070 RGB8 |
| Algorithm | Separable Lanczos-3 |
| Horizontal pass | AVX2, 8-lane multiplication and horizontal reduction |
| Vertical pass | AVX2/FMA, eight output pixels per iteration |
| Threads | Single-threaded |
| Benchmark | Google Benchmark |
| Repetitions | 10 |
| Build | `build/profile` |
| Memory management | Preallocated scratch and output buffers |

The latest implementation performs both interpolation passes using AVX2 intrinsics. Coefficient initialization and scratch-buffer allocation occur outside the resize hot path.

### AVX2 Benchmark Results

| Metric | Fixed 8-iteration vertical loop | Variable-bound vertical loop |
|---|---:|---:|
| Single-plane median | 1.7169 ms | **1.6619 ms** |
| Single-plane mean | 1.7192 ms | 1.6738 ms |
| Single-plane standard deviation | 0.0260 ms | 0.0272 ms |
| Single-plane CV | 1.51% | 1.62% |
| RGB median | 5.1879 ms | **4.9913 ms** |
| RGB mean | 5.2025 ms | 5.0243 ms |
| RGB standard deviation | 0.0483 ms | 0.0593 ms |
| RGB CV | 0.93% | 1.18% |

The variable-bound vertical loop improved the RGB median from 5.1879 ms to 4.9913 ms.

**Observed improvement: approximately 3.79%.**

The variable-bound implementation was retained as the current best-performing version.

These measurements were collected in separate benchmark runs. Additional interleaved repetitions would help establish whether the improvement persists under controlled conditions.

### Historical Performance Progression

| Implementation | RGB execution time | Speedup vs initial 2D implementation |
|---|---:|---:|
| Naive direct 2D Lanczos-3 | 1896.13 ms | 1.0× |
| Separable scalar implementation | 22.896 ms | 82.8× |
| Optimized scalar with precomputed coefficients | 16.501 ms | 114.9× |
| AVX2 vertical pass, scalar horizontal | 10.648 ms | 178.1× |
| AVX2 horizontal + vertical, fixed vertical loop | 5.188 ms | 365.5× |
| **AVX2 horizontal + vertical, variable-bound loop** | **4.991 ms** | **379.9×** |

The historical measurements are not all directly comparable. They use different implementations, benchmarking scopes, and memory-allocation strategies.

The approximately **3.31× improvement over the 16.501 ms optimized scalar baseline** is therefore indicative rather than an isolated measurement of SIMD speedup.

### Application-Level Pass Timings

A separate application run measured the individual horizontal and vertical passes.

The first invocation was a warm-up and is excluded from the following table.

| Channel | Horizontal pass | Vertical pass | Total |
|---|---:|---:|---:|
| R | 0.894 ms | 1.211 ms | 2.105 ms |
| G | 0.916 ms | 1.126 ms | 2.042 ms |
| B | 0.878 ms | 1.207 ms | 2.084 ms |
| **RGB total** | **2.688 ms** | **3.544 ms** | **6.232 ms** |

Reported application kernel time: **6.238 ms**.

Approximate execution-time distribution:

- Horizontal pass: 43.1%
- Vertical pass: 56.9%

Individual pass measurements showed variability and should not be interpreted as precise isolated kernel throughput.

The application-level measurement and Google Benchmark use different execution conditions; their timings are reported separately.

## Optimization Log

### 1. Replace direct 2D filtering with two separable passes

**Observation:** Direct 2D Lanczos-3 combines horizontal and vertical weights, typically considering ~6 × 6 = 36 neighboring pixels per interior destination pixel. The kernel is separable.

**Change:** First resample each source row horizontally into a floating-point scratch buffer; then resample the scratch buffer vertically and convert to `uint8_t` only at the end.

```text

Source RGB plane       x × y             uint8_t

        |

        | Horizontal Lanczos-3

        v

Scratch buffer         x × res_y         float

        |

        | Vertical Lanczos-3

        v

Output RGB plane       res_x × res_y     uint8_t

```

For 2× upscaling, direct 2D filtering takes roughly `144*x*y` weighted contributions per plane; the separable version takes roughly `36*x*y` (ignoring border effects). On the M1 Pro the observed total time dropped from ~1,100 ms to ~250 ms (~4.4×).

### 2. Improve vertical traversal and reuse vertical weights

**Observation:** With output columns outermost, successive destination writes jump by a row stride. For a fixed output row, the same vertical Lanczos weights apply to every output column.

**Changes on M1 Pro:** Switching to output-row-major traversal reduced ~250 ms to 244.43 ms; calculating the vertical weights once per destination row reduced it further to 98.10 ms. Reusing weights avoids repeated `sin()` evaluations.

### 3. Reuse horizontal weights across source rows

**Observation:** Horizontal weights depend on the destination column, not the source row.

**Change:** Make destination columns the outer loop and reuse ~6–7 weights for all source rows. The resulting Intel application run reached 22.8961 ms for RGB, down from 1,896.13 ms for the initial direct-2D implementation. This speedup reflects **all preceding optimizations**, not just the horizontal loop change.

**Trade-off:** Column-major horizontal processing writes a strided float scratch buffer, but gives excellent temporal locality for the current column's coefficients.

### 4. Precompute normalized coefficient tables and share across RGB

**Change:** `lanczos_init(x, y)` builds two flattened coefficient tables: one for each output column (`res_y × (2*window_sz + 1)`) and one for each output row (`res_x × (2*window_sz + 1)`). Boundary-clipped coefficients are stored starting at index 0 for the coordinate's first valid neighbor. Each coordinate's coefficients are normalized **during initialization**; the same tables serve R, G, and B.

**Benefit:** Avoid repeated trigonometric calculations and remove per-pixel weight-sum accumulation/division. Initialization is amortized across the three channels or across repeated resizes of the same geometry. Float-rounding differences from the earlier version are possible.

**Experiment:** Tried horizontal output-row-major traversal while retaining precomputed coefficients to make scratch-buffer writes contiguous. Despite better sequential stores, it was slower in exploratory application runs (~34.5 ms RGB total). The likely trade-off is repeated coefficient-table access and additional loop/indexing work; no isolated cache-miss measurement established the cause. Reverted to the faster output-column-major horizontal traversal.

### 5. Profile and control measurement variability

`perf` on the Intel system initially showed misleading startup/library activity (including `rand()` in the synthetic-input benchmark). After using deterministic input and profiling user-space cycles without call graphs, the P-core profile attributed approximately **97.7% of sampled cycles to `lanczos_resize()`** and less than ~1% to `\_\_sinf_fma`. This supported focusing on the multiply-accumulate loops and data access rather than additional transcendental optimizations.

On the **hybrid P/E-core i9-13980HX**, application kernel time varied substantially between invocations and the first RGB channel was often slower. CPU 8 was identified as a P-core (logical CPUs 0–15 are P-cores; 16–31 are E-cores). Pinning the application alone did not eliminate the first-call penalty. Adding a warm-up resize before the measured RGB sequence produced six pinned application times:

| Warmed-up pinned application run | RGB kernel time |
|---|---:|
| 1 | 15.416 ms |
| 2 | 15.328 ms |
| 3 | 15.474 ms |
| 4 | 15.443 ms |
| 5 | 15.632 ms |
| 6 | 15.466 ms |

The warm-up invocation still fluctuated substantially. **Likely factors** include CPU frequency ramp-up, cache warming, and minor page faults from first-touch memory allocation; their individual contributions were **not conclusively measured**. Two un-warmed pinned `perf stat` runs reported 6,590 minor faults and zero major faults across the entire application, which does not isolate the resize kernel.

**Practical decision:** For SIMD work, use steady-state comparisons with consistent CPU pinning, warm-up, allocations, compilation, and benchmark scope. Preserve cold-start behavior as a separate concern rather than interpreting it as scalar-kernel throughput.

### 6. First AVX2 vertical-pass result

Implemented an AVX2/FMA vertical pass and recorded a pinned release application kernel time of **10.6482 ms**, with individual vertical-pass logs of **1.78825, 1.16033, 0.725781, and 0.785632 ms**. See the latest application run above for measurement caveats. Horizontal resampling has not yet been SIMD-optimized.

### 7. AVX2 Horizontal Interpolation

Implemented horizontal interpolation using AVX2 with one output pixel calculated per SIMD reduction.

For each output pixel:

- Load up to seven source pixels from contiguous image memory.
- Convert unsigned 8-bit pixels into 32-bit floating-point values.
- Load the corresponding precomputed interpolation coefficients.
- Perform eight-lane floating-point multiplication.
- Horizontally reduce the eight contributions into one output value.

The coefficient table uses a stride of eight floats per output coordinate, with unused entries initialized to zero.

This approach avoids full-image transposition and gather instructions.

### 8. AVX2 Vertical Interpolation

Implemented AVX2/FMA interpolation across eight adjacent output columns.

Eight output pixels are accumulated simultaneously using shared interpolation coefficients and contiguous scratch-buffer loads.

After accumulation, values are converted from floating point into packed unsigned 8-bit output pixels.

Compared two vertical-loop strategies:

- Fixed eight-iteration loop using zero-padded coefficients and scratch memory.
- Variable-bound loop that processes only valid source neighbors.

The variable-bound implementation achieved approximately **3.79% lower RGB median latency** in the recorded benchmark runs.

### 9. Moving Memory Allocations Outside the Hot Path

Refactored the resize interface to accept preallocated output buffers.

Scratch-buffer allocation and coefficient initialization are performed once and reused across consecutive resize calls.

The resize kernel no longer allocates or frees its primary working buffers during execution.

This reduces allocation overhead and establishes a reusable workspace for future image and video processing experiments.

The resulting benchmark measures the steady-state interpolation kernel, excluding initialization and working-buffer allocation.

### 10. Current Best Performance

**AVX2 Lanczos-3, Intel Core i9-13980HX**

- Input: 736 × 535 RGB8
- Output: 1472 × 1070 RGB8
- Single-plane median: **1.662 ms**
- Three-plane RGB median: **4.991 ms**
- RGB benchmark CV: **1.18%**
- Historical improvement over initial naive implementation: approximately **380×**

Further optimization will focus on profiling instruction costs, memory access patterns, and alternative SIMD strategies.

## Resampling Mathematics

### 1. Destination-to-source coordinate mapping

Map destination pixel centers back into source coordinates:

$$
c_s = (c_d + 0.5)\frac{W_s}{W_d} - 0.5
$$

$$
r_s = (r_d + 0.5)\frac{H_s}{H_d} - 0.5
$$

Here, $W$ denotes width and $H$ denotes height.

The implementation names its **row** dimension `x` and **column** dimension `y`.

### 2. Lanczos-3 kernel

Normalized sinc:

$$
\mathrm{sinc}(t) =
\begin{cases}
1, & t = 0 \\
\frac{\sin(\pi t)}{\pi t}, & t \neq 0
\end{cases}
$$

Lanczos-3 kernel:

$$
L_3(t) =
\begin{cases}
\mathrm{sinc}(t)\mathrm{sinc}(t/3), & |t| < 3 \\
0, & |t| \geq 3
\end{cases}
$$

### 3. Direct 2D interpolation

For source row $i$ and column $j$, the combined weight is:

$$
w(i,j) = L_3(r_s-i)L_3(c_s-j)
$$

The interpolated destination pixel is:

$$
D(r_d,c_d) =
\frac{\sum_{i,j} S(i,j)w(i,j)}
{\sum_{i,j}w(i,j)}
$$

Only source pixels within the Lanczos support contribute. An interior output pixel considers approximately **6 × 6 = 36** candidates.

### 4. Separable interpolation

First, horizontal filtering:

$$
T(i,c_d) =
\frac{\sum_j S(i,j)L_3(c_s-j)}
{\sum_j L_3(c_s-j)}
$$

Then, vertical filtering:

$$
D(r_d,c_d) =
\frac{\sum_i T(i,c_d)L_3(r_s-i)}
{\sum_i L_3(r_s-i)}
$$

The normalization factors multiply because:

$$
\sum_{i,j}w_x(i)w_y(j) =
\left(\sum_i w_x(i)\right)
\left(\sum_j w_y(j)\right)
$$

With consistent boundary handling and no intermediate 8-bit quantization, the two-pass calculation is mathematically equivalent to direct 2D filtering. Floating-point evaluation order can produce small numerical differences.

### 5. Output conversion

Keep the horizontal intermediate as `float`, and only after vertical interpolation convert to 8 bits:

$$
D_{\mathrm{8bit}} =
\mathrm{clamp}(\mathrm{trunc}(D),0,255)
$$

This reflects the implementation's integer cast (truncation), rather than rounding to nearest.

Keep the horizontal intermediate as `float`, and only after vertical interpolation convert to 8 bits:

$$

D\_{*\mathrm*{8bit}} = *\mathrm*{clamp}(*\mathrm*{trunc}(D), 0, 255)

$$

This reflects the implementation's integer cast (truncation), rather than rounding to nearest.

## Building

The project uses CMake, Google Benchmark, and local `stb_image` / `stb_image_write` headers.

```bash

*# Debug (ASan is enabled in Debug when ENABLE_ASAN=ON)*

cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON

cmake --build build/debug -j

*# Release*

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=OFF

cmake --build build/release -j

*# Execute from the repository root*

./build/release/lanczos-main

./build/release/lanczos-bench

```

The application currently uses hard-coded image paths. Debug-level logging is disabled in Release, while INFO logging may remain enabled. The application writes a PNG output; verify its filename in the current source if necessary.

## Next Steps

- [x] Naive direct 2D Lanczos-3 baseline

- [x] Separable scalar implementation using a float scratch buffer

- [x] Vertical row-major traversal and per-coordinate coefficient reuse

- [x] Shared normalized coefficient tables across RGB planes

- [x] Compare horizontal column-major and row-major loop orders; retain the faster scalar order

- [x] Add Google Benchmark and investigate CPU performance using Linux `perf`

- [x] Identify P/E-core scheduling and warm-up as material benchmarking concerns

- [x] Visually validate current output

- [x] **AVX2 vertical pass**, processing eight adjacent output columns with shared coefficients and contiguous float loads

- [x] Optimize / vectorize the horizontal pass and benchmark against the frozen scalar version

- [ ] CUDA implementation and CPU–GPU comparisons

- [ ] Optional later: quantitative pixel-difference, MAE, or PSNR correctness testing

## Scope

**Currently supported/tested:** RGB8, planar processing, separable scalar CPU Lanczos-3 with an AVX2/FMA vertical pass, 2× upscale, and the sample 736 × 535 image. The original direct-2D version is a historical reference implementation.

**Not yet implemented:** General-scale resizing, 16-bit channels, alpha-aware filtering, downscale-specific antialiasing, AVX2 horizontal pass, NEON, and CUDA.

New measurements should be **appended**, with platform, dimensions, build flags, CPU affinity, warm-up, benchmark scope, and measurement method documented. Historical measurements should not be silently replaced by newer results.
