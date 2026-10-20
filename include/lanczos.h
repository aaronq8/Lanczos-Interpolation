#pragma once

#include <cstdint>
#include <utility>
#define PI 3.14159265f

constexpr int window_sz = 3;
constexpr float scale = 0.5;
constexpr int scale_rec = 2;
constexpr float epsilon = 1e-8;

std::pair<float *, float *> lanczos_init(int, int);
uint8_t *lanczos_resize(uint8_t *, int, int, float *, float *);