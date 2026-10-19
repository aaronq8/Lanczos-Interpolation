#pragma once

#include <cstdint>
#define PI 3.14159265f

constexpr int window_sz = 3;
constexpr float scale = 0.5;
constexpr int scale_rec = 2;

uint8_t *lanczos_resize(uint8_t *, int, int);