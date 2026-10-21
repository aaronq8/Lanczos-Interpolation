# Lanczos Image Resampling

README is AI Generated, Code is handwritten.

Goal: implement and optimize **Lanczos-3 image resampling**, starting with a straightforward scalar CPU implementation and progressing toward CPU SIMD (AVX2 / possibly NEON) and CUDA. The emphasis is on interpolation mathematics, memory layout, profiling, and measurable performance improvements—not on calling a prebuilt resize library.

## Visual Comparison

### Original Image (736 × 535)

![Original Image](sample-images/0.jpg)

### Lanczos-3 Resized (1472 × 1070)

![Lanczos-3 Resized](sample-images/resized-sep.png)

## Current Status

**Scalar CPU implementation complete; AVX2 vertical pass implemented and running. Horizontal SIMD is next.**

- Loads JPEG images with `stb_image`, requesting three RGB8 channels; splits interleaved RGB into planar R, G, and B.

- Resizes each plane independently with separable Lanczos-3 (horizontal followed by vertical).

- Uses pixel-center destination-to-source mapping, clamped source-neighbor bounds, and a floating-point intermediate buffer; final values are truncated and clamped to `[0, 255]`.

- Builds **normalized coefficient tables once per image geometry**, with one set of weights for each destination column (horizontal) and each destination row (vertical). Shares those tables across all three color planes.

- Uses **destination-column-major traversal for the horizontal pass** to maximize coefficient reuse; **destination-row-major traversal for the vertical pass** to write contiguous output pixels.

- Uses Google Benchmark and Linux `perf` to evaluate the implementation. Benchmarking on the hybrid-core Intel CPU requires controlling core affinity and warm-up.

- Visually checked the generated image against the previous implementation. A quantitative pixel-difference/PSNR comparison is intentionally deferred; visual similarity is not a formal numerical-equivalence test.

**Implementation convention:** `x = rows (height)` and `y = columns (width)`; planar memory uses `img[row * width + col]`.

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

### Latest AVX2 vertical-pass application run (single invocation)

The horizontal pass remains scalar; the vertical pass uses AVX2/FMA for eight adjacent output columns, with a scalar tail. These are **application timer logs**, not Google Benchmark statistics.

| Parameter | Recorded value |
|---|---|
| CPU | Intel Core i9-13980HX (Ubuntu) |
| Input / output | 736 × 535 RGB8 → 1472 × 1070 RGB8 |
| Build / command | `build/release/lanczos-main`, `taskset -c 8` |
| Implementation | Precomputed coefficients; scalar horizontal pass + AVX2/FMA vertical pass |
| **Total reported Lanczos kernel** | **10.6482 ms** |

Vertical-pass timer output **in the order logged** (the application log does not label channels or warm-up):

| Logged vertical pass | Time |
|---|---:|
| 1 | 1.78825 ms |
| 2 | 1.16033 ms |
| 3 | 0.725781 ms |
| 4 | 0.785632 ms |

There are four vertical-pass lines, consistent with a possible warm-up followed by three RGB planes, **but the log alone does not identify which invocation is which**. Do not sum all four and report it as the three-plane vertical time. The single total kernel measurement should not be treated as a controlled speedup against the earlier scalar Google Benchmark medians; run matched repeated benchmarks to establish a stable comparison. The reported total excludes image loading and PNG writing.

## Historical Performance Measurements

These are exploratory measurements made during development. They are retained as an optimization log, **not** presented as identical, controlled benchmark runs.

| Platform | Implementation / change | Measured total RGB kernel time | Speedup vs original same-machine naive baseline |
|---|---|---:|---:|
| Intel Core i9-13980HX (Ubuntu) | Naive direct 2D Lanczos-3 | **1,896.13 ms** | 1.00× |
| Intel Core i9-13980HX (Ubuntu) | Separable + coefficients reused within passes | **22.8961 ms** | ~82.8× |
| Intel Core i9-13980HX (Ubuntu) | Shared, normalized tables; steady-state application after warm-up | **~15.3–15.6 ms** | ~121–124× (indicative only) |
| Intel Core i9-13980HX (Ubuntu) | Scalar horizontal + AVX2 vertical (single pinned application run) | **10.6482 ms** | ~178× (indicative only; not a controlled speedup) |
| Apple M1 Pro (macOS, Apple Clang) | Naive direct 2D Lanczos-3 | **~1,100 ms** | 1.00× |
| Apple M1 Pro (macOS, Apple Clang) | Initial separable scalar | **~250 ms** | ~4.4× |
| Apple M1 Pro (macOS, Apple Clang) | Vertical row-major traversal | **244.43 ms** | ~4.5× |
| Apple M1 Pro (macOS, Apple Clang) | Precompute vertical weights per output row | **98.10 ms** | ~11.2× |

The original Intel naive baseline, older Intel 22.8961 ms run, and M1 Pro figures were measured with application timers or single runs; the more recent Google Benchmark measurements use multiple repetitions. **Do not compare absolute performance across the Intel and Apple systems or quote the historical speedup as a controlled benchmark result.**

Historical Intel 22.8961 ms pass breakdown:

| Channel | Horizontal | Vertical |
|---|---:|---:|
| R | 5.8977 ms | 5.4353 ms |
| G | 2.2811 ms | 3.6287 ms |
| B | 2.1836 ms | 3.2636 ms |
| **Sum** | **10.3624 ms** | **12.3276 ms** |

The first channel was notably slower, motivating investigation of cold-start effects.

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

## Resampling Mathematics

### 1. Destination-to-source coordinate mapping

Map destination pixel centers back into source coordinates:

$$

c_s = (c_d + 0.5)*\frac*{W_s}{W_d} - 0.5

$$

$$

r_s = (r_d + 0.5)*\frac*{H_s}{H_d} - 0.5

$$

Here `W` denotes width and `H` height; the implementation names its **row** dimension `x` and **column** dimension `y`.

### 2. Lanczos-3 kernel

Normalized sinc:

$$

*\mathrm*{sinc}(t) =

*\begin*{cases}

1, & t = 0 \\\\

*\frac*{\sin(\pi t)}{\pi t}, & t \neq 0

*\end*{cases}

$$

Lanczos-3 kernel:

$$

L_3(t) =

*\begin*{cases}

*\mathrm*{sinc}(t)*\mathrm*{sinc}(t/3), & |t| < 3 \\\\

0, & |t| \geq 3

*\end*{cases}

$$

### 3. Direct 2D interpolation

For source row `i` and column `j`, the combined weight is:

$$

w(i,j) = L_3(r_s-i)L_3(c_s-j)

$$

$$

D(r_d,c_d) = *\frac*{*\sum\_*{i,j}S(i,j)w(i,j)}{*\sum\_*{i,j}w(i,j)}

$$

Only source pixels within the Lanczos support contribute; an interior output pixel considers approximately **6 × 6 = 36** candidates.

### 4. Separable interpolation

First, horizontal filtering:

$$

T(i,c_d) = *\frac*{\sum_j S(i,j)L_3(c_s-j)}{\sum_j L_3(c_s-j)}

$$

Then, vertical filtering:

$$

D(r_d,c_d) = *\frac*{\sum_i T(i,c_d)L_3(r_s-i)}{\sum_i L_3(r_s-i)}

$$

The normalization factors multiply because:

$$

*\sum\_*{i,j}w_x(i)w_y(j) = \left(\sum_i w_x(i)\right)\left(\sum_j w_y(j)\right)

$$

With consistent boundary handling and no intermediate 8-bit quantization, the two-pass calculation is mathematically equivalent to direct 2D filtering. Floating-point evaluation order can produce small numerical differences.

### 5. Output conversion

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

- [ ] Optimize / vectorize the horizontal pass and benchmark against the frozen scalar version

- [ ] Optional ARM NEON implementation

- [ ] CUDA implementation and CPU–GPU comparisons

- [ ] Optional later: quantitative pixel-difference, MAE, or PSNR correctness testing

## Scope

**Currently supported/tested:** RGB8, planar processing, separable scalar CPU Lanczos-3 with an AVX2/FMA vertical pass, 2× upscale, and the sample 736 × 535 image. The original direct-2D version is a historical reference implementation.

**Not yet implemented:** General-scale resizing, 16-bit channels, alpha-aware filtering, downscale-specific antialiasing, AVX2 horizontal pass, NEON, and CUDA.

New measurements should be **appended**, with platform, dimensions, build flags, CPU affinity, warm-up, benchmark scope, and measurement method documented. Historical measurements should not be silently replaced by newer results.
