# Lanczos Image Resampling

README is AI Generated, Code is handwritten

Goal is to implement & optimize **Lanczos-3 image resampling**, starting with a straightforward scalar CPU implementation and progressing toward CPU SIMD and CUDA versions.

The emphasis is on understanding interpolation, numerical behavior, memory layout, profiling, and performance optimization—not on using a prebuilt resize implementation.

## Current status

**Working baseline: naive, direct 2D Lanczos-3 resampler (CPU).**

- Loads JPEG images using `stb_image` (requests 8-bit, three-channel RGB output).
- Splits interleaved RGB into three planar 8-bit channels.
- Resizes each channel independently with a scalar Lanczos-3 kernel.
- Uses pixel-center coordinate mapping and a 2D product of horizontal and vertical Lanczos weights.
- Normalizes the accumulated weighted sum and clamps output values to `[0, 255]`.
- Interleaves the resized RGB planes and writes a PNG using `stb_image_write`.
- Produces a visually correct resized image in the current test.

**Implementation notes:** The baseline directly evaluates the 2D kernel for every destination pixel; Image dimensions in the implementation use `x = rows (height)` and `y = columns (width)`.

## Performance log

Measured on an **i9-13980HX 5.6 ghz 2MB L1 32 MB L2** machine, using the **Release** executable:

```text
$ ./build/release/lanczos
[INFO] Loading image from sample-images....
[INFO] Loaded image....x : 535, y : 736, channels : 3
[INFO] Lanczos Kernel elapsed : 1896.13 ms
[INFO] writing png...
```

| Metric | Baseline |
|---|---:|
| Input image | `sample-images/0.jpg` |
| Input resolution | 736 × 535 (width × height) |
| Input channels | RGB8 (3) |
| Lanczos window radius | 3 |
| Implementation | Scalar CPU, direct 2D, three planar channels |
| Build | Release (`./build/release/lanczos`) |
| Measured kernel time | **1,896.13 ms (~1.90 s)** |


**Scaling configuration:** The current source implementation has used a **2× resize in each dimension** (`scale = 0.5`, `scale_rec = 2`).

## Resampling math

Destination pixels map back to source coordinates via pixel-center alignment:

\[
x_s = (x_d + 0.5)\frac{W_s}{W_d} - 0.5
\]

(and equivalently for the vertical axis).

The normalized sinc function is

\[
\operatorname{sinc}(t) =
\begin{cases}
1 & t=0\\
\frac{\sin(\pi t)}{\pi t} & t\ne0
\end{cases}
\]

The Lanczos-3 kernel is

\[
L_3(t) =
\begin{cases}
\operatorname{sinc}(t)\operatorname{sinc}(t/3) & |t| < 3\\
0 & |t| \ge 3
\end{cases}
\]

For a single color plane, a destination sample is computed from neighboring source samples:

\[
D(x_d,y_d) =
\frac{\sum_{i,j} S(i,j)L_3(x_s-i)L_3(y_s-j)}
{\sum_{i,j} L_3(x_s-i)L_3(y_s-j)}.
\]

Only in-bounds source pixels within the radius are included. Current implementation uses floating-point accumulators and converts the final clamped value back to 8 bits.

## Building

The project uses CMake and local `stb_image` / `stb_image_write` headers.

```bash
# Debug build (ASan can be enabled with the project's ENABLE_ASAN option)
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j

# Optimized Release build
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=OFF
cmake --build build/release -j

# Run from the repository root (image path is currently relative)
./build/release/lanczos
```

The current executable uses `sample-images/0.jpg` as input and writes `sample-images/resized.png`. Paths are currently hard-coded.

> **Logging note:** The recorded Release run still printed `[INFO]` messages. If the intended policy is *no logs in Release*, check that the logger macros and `DISABLE_LOGS` compile definition are active, then rebuild.


## Scope

- **Current:** RGB8 images; scalar CPU; Lanczos-3; 2× upscale test.
- **Not yet implemented:** arbitrary scaling ratios, 16-bit channels, alpha-aware filtering, explicit antialiasing for downscaling, AVX2, CUDA.

This README is a snapshot of the current code; future measurements should be appended with their exact test settings rather than replacing the original result.
