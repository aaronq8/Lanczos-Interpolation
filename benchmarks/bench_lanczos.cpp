#include "lanczos.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>

// benchmarking for 1 plane
static void BM_lanczosSinglePlane(benchmark::State &state) {
  constexpr int x = 535, y = 736;
  uint8_t *channel = (uint8_t *)std::malloc(x * y);
  for (int i = 0; i < x * y; i++)
    channel[i] = (uint8_t)(i % 256);
  auto [precalc_l_y, precalc_l_x] = lanczos_init(x, y);

  for (auto _ : state) {
    uint8_t *channel_out =
        lanczos_resize(channel, x, y, precalc_l_y, precalc_l_x);
    benchmark::DoNotOptimize(channel_out);
    std::free(channel_out);
  }
  std::free(channel);
  std::free(precalc_l_y);
  std::free(precalc_l_x);
}

static void BM_lanczosRGB(benchmark::State &state) {
  constexpr int x = 535, y = 736;
  uint8_t *channel_r = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_g = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_b = (uint8_t *)std::malloc(x * y);

  for (int i = 0; i < x * y; i++) {
    channel_r[i] = (uint8_t)(i % 256);
    channel_g[i] = (uint8_t)(i % 256);
    channel_b[i] = (uint8_t)(i % 256);
  }

  auto [precalc_l_y, precalc_l_x] = lanczos_init(x, y);

  for (auto _ : state) {
    uint8_t *r_out = lanczos_resize(channel_r, x, y, precalc_l_y, precalc_l_x);
    uint8_t *g_out = lanczos_resize(channel_g, x, y, precalc_l_y, precalc_l_x);
    uint8_t *b_out = lanczos_resize(channel_b, x, y, precalc_l_y, precalc_l_x);

    benchmark::DoNotOptimize(r_out);
    benchmark::DoNotOptimize(g_out);
    benchmark::DoNotOptimize(b_out);

    std::free(r_out);
    std::free(g_out);
    std::free(b_out);
  }
  std::free(channel_r);
  std::free(channel_g);
  std::free(channel_b);
  std::free(precalc_l_y);
  std::free(precalc_l_x);
}

BENCHMARK(BM_lanczosSinglePlane);
BENCHMARK(BM_lanczosRGB);