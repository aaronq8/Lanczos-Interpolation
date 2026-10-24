#include "lanczos.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>

// benchmarking for 1 plane
static void BM_lanczosSinglePlane(benchmark::State &state) {
  constexpr int x = 535, y = 736, sz = x * y + 2 * window_sz + 1;
  const int res_x = x * scale_rec;
  const int res_y = y * scale_rec;

  uint8_t *channel = (uint8_t *)std::malloc(sz);
  uint8_t *res_w = (uint8_t *)std::malloc(res_x * res_y);

  for (int i = 0; i < sz; i++)
    channel[i] = (uint8_t)(i % 256);
  auto [precalc_l_y, precalc_l_x] = lanczos_init(x, y);

  lanczos_resize(channel, x, y, precalc_l_y, precalc_l_x, &res_w);

  for (auto _ : state) {
    lanczos_resize(channel, x, y, precalc_l_y, precalc_l_x, &res_w);
  }

  lanczos_free();
  std::free(res_w);
  std::free(channel);
  std::free(precalc_l_y);
  std::free(precalc_l_x);
}

static void BM_lanczosRGB(benchmark::State &state) {
  constexpr int x = 535, y = 736;
  const int res_x = x * scale_rec;
  const int res_y = y * scale_rec;

  uint8_t *channel_r = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_g = (uint8_t *)std::malloc(x * y);
  uint8_t *channel_b = (uint8_t *)std::malloc(x * y);
  uint8_t *res_w = (uint8_t *)std::malloc(res_x * res_y);

  for (int i = 0; i < x * y; i++) {
    channel_r[i] = (uint8_t)(i % 256);
    channel_g[i] = (uint8_t)(i % 256);
    channel_b[i] = (uint8_t)(i % 256);
  }

  auto [precalc_l_y, precalc_l_x] = lanczos_init(x, y);
  lanczos_resize(channel_r, x, y, precalc_l_y, precalc_l_x, &res_w);

  for (auto _ : state) {
    lanczos_resize(channel_r, x, y, precalc_l_y, precalc_l_x, &res_w);
    lanczos_resize(channel_g, x, y, precalc_l_y, precalc_l_x, &res_w);
    lanczos_resize(channel_b, x, y, precalc_l_y, precalc_l_x, &res_w);
  }

  lanczos_free();
  std::free(res_w);
  std::free(channel_r);
  std::free(channel_g);
  std::free(channel_b);
  std::free(precalc_l_y);
  std::free(precalc_l_x);
}

BENCHMARK(BM_lanczosSinglePlane);
BENCHMARK(BM_lanczosRGB);