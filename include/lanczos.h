#pragma once

#include <cstdint>
#include <utility>
#define PI 3.14159265f

constexpr int window_sz = 3;
constexpr float scale = 0.5;
constexpr int scale_rec = 2;
constexpr float epsilon = 1e-8;
// m256 loads 8 values, even though at-max we can have 7 weights
constexpr int l_precalc_stride = 2 * window_sz + 2;

std::pair<float *, float *> lanczos_init(int, int);
void lanczos_resize(uint8_t *, int, int, float *, float *, uint8_t **);
void lanczos_free();