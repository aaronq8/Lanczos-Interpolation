#include "lanczos.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>

// benchmarking for 1 plane
static void BM_lanczosSinglePlane(benchmark::State &state) {
  constexpr int x = 535, y = 736;
  uint8_t *channel = (uint8_t *)std::malloc(x * y);
  for (int i = 0; i < x * y; i++)
    channel[i] = (uint8_t)(rand() % 256);

  for (auto _ : state) {
    uint8_t *channel_out = lanczos_resize(channel, x, y);
    benchmark::DoNotOptimize(channel_out);
    std::free(channel_out);
  }
  std::free(channel);
}

static void BM_lanczosRGB(benchmark::State &state) {
  constexpr int x = 535, y = 736;
  uint8_t *channel_r = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_g = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_b = (uint8_t *)std::malloc(x * y);

  for (int i = 0; i < x * y; i++) {
    channel_r[i] = (uint8_t)(rand() % 256);
    channel_g[i] = (uint8_t)(rand() % 256);
    channel_b[i] = (uint8_t)(rand() % 256);
  }

  for (auto _ : state) {
    uint8_t *r_out = lanczos_resize(channel_r, x, y);
    uint8_t *g_out = lanczos_resize(channel_g, x, y);
    uint8_t *b_out = lanczos_resize(channel_b, x, y);

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
}

BENCHMARK(BM_lanczosSinglePlane);
BENCHMARK(BM_lanczosRGB);