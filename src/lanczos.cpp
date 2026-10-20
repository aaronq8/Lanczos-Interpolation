#include "lanczos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "logger.h"

inline float sinc(float x) {
  // sinc(1) ~ -2*10^-8
  if (std::abs(x) < epsilon)
    return 1.0;
  return std::sin(PI * x) / (PI * x);
}

inline float L3(float x) {
  // Lanczos with window size=3
  return sinc(x) * sinc(x / (1.0f * window_sz));
}

/*
 * we can precalc L3 values required for y interpolation for each column for
 * horizontal pass and precalc L3 values required for x interpolation for each
 * row for vertical pass
 * precalc_l_y -> (col0)[w0...w6](col1)[w0...w6]
 * precalc_l_x -> (row0)[w0...w6](row1)[w0...w6]
 */
std::pair<float *, float *> lanczos_init(int x, int y) {
  int res_x = scale_rec * x;
  int res_y = scale_rec * y;

  float *precalc_l_y =
      (float *)std::malloc(res_y * (2 * window_sz + 1) * sizeof(float));
  float *precalc_l_x =
      (float *)std::malloc(res_x * (2 * window_sz + 1) * sizeof(float));

  for (int ry = 0; ry < res_y; ry++) {
    float y_src = ((ry + 0.5) * scale) - 0.5;
    // valid neighbours [rx][y_min....y_max]
    int y_min = std::max((int)ceil(y_src) - window_sz, 0);
    int y_max = std::min((int)floor(y_src) + window_sz, y - 1);
    int col_num = ry * (2 * window_sz + 1);
    float tot_wt_y = 0.0f;
    for (int i = y_min; i <= y_max; i++) {
      //[w0,w1,...w6][w0,...w6]...
      precalc_l_y[col_num + i - y_min] = L3(y_src - i);
      tot_wt_y += precalc_l_y[col_num + i - y_min];
    }
    // normalize
    for (int i = y_min; i <= y_max; i++) {
      precalc_l_y[col_num + i - y_min] /= tot_wt_y;
    }
  }

  for (int rx = 0; rx < res_x; rx++) {
    float x_src = ((rx + 0.5) * scale) - 0.5;
    // valid neighbours [x_min.....x_max][ry]
    int x_min = std::max(0, (int)ceil(x_src) - window_sz);
    int x_max = std::min((int)floor(x_src) + window_sz, x - 1);
    float tot_wt_x = 0.0f;
    // precalculate L3 values
    // x_min.....x_src.....x_max
    // ~50ms -> ~5ms => 900% faster
    int row_num = rx * (2 * window_sz + 1);
    for (int i = x_min; i <= x_max; i++) {
      precalc_l_x[row_num + i - x_min] = L3(x_src - i);
      tot_wt_x += precalc_l_x[row_num + i - x_min];
    }
    // normalize
    for (int i = x_min; i <= x_max; i++) {
      precalc_l_x[row_num + i - x_min] /= tot_wt_x;
    }
  }

  return {precalc_l_y, precalc_l_x};
}

/*
we will fix scale (width_src/width_dest) = 0.5 and lanczos window size = 3
L3(x) = sinc(x) * sinc(x/3), |x| < 3
x here is dist b/w dest pixel mapped to source and neighbouring source pixels

mapping dest pixel to src
x_src = (x_dest + 0.5)*scale - 0.5
we add 0.5 to ensure dist(1st dest,1st dest mapped src) == (lst dest,lst dest
mapped src)...else dest image will be shifted left
eg scale = 0.5 and 4 pixels in src
0 -> -0.25, 7->3.25 ..... |-0.25| == |0.25|
*/
uint8_t *lanczos_resize(uint8_t *img, int x, int y, float *precalc_l_y,
                        float *precalc_l_x) {
  LOG_DEBUG("resizing....");

  int res_x = scale_rec * x;
  int res_y = scale_rec * y;
  uint8_t *res = (uint8_t *)std::malloc(res_x * res_y);
  /*
   * We will Linearly Seperate the Lanczos Kernel
   * Interpolate in y and store values in temp scratch buffer with row x scale *
   * width then Interpolate in x, this reduces number of L3 calls from ~36 -> ~6
   * per pixel (window_sz = 3 and interior pixels)
   */
  float *buf = (float *)std::malloc(res_y * x * sizeof(float));
  // horizontal pass
  // we go column major so that y_src is fixed
  auto start_hz = std::chrono::steady_clock::now();
  for (int ry = 0; ry < res_y; ry++) {
    float y_src = ((ry + 0.5) * scale) - 0.5;
    // valid neighbours [rx][y_min....y_max]
    int y_min = std::max((int)ceil(y_src) - window_sz, 0);
    int y_max = std::min((int)floor(y_src) + window_sz, y - 1);
    for (int rx = 0; rx < x; rx++) {
      int px = rx * res_y + ry;
      float buf_val = 0.0;
      for (int j = y_min; j <= y_max; j++) {
        // calculate weight for neighbour rx,j
        // L(0)=1 so we can ignore row/x weight
        //[w0....w6][w0...w6]...
        float wt = precalc_l_y[(ry * (2 * window_sz + 1)) + j - y_min];
        buf_val += (wt * (float)img[rx * y + j]);
      }
      // wieghts were normalized during precalc
      buf[px] = buf_val;
    }
  }

  auto end_hz = std::chrono::steady_clock::now();

  double elapsed_ms_hz =
      std::chrono::duration<double, std::milli>(end_hz - start_hz).count();

  LOG_INFO("horizontal pass : " << elapsed_ms_hz << " ms");

  auto start_vert = std::chrono::steady_clock::now();

  // vertical pass
  // row-wise so that res access is cache aligned => ~4% faster
  // buf access still needs to be improved
  for (int rx = 0; rx < res_x; rx++) {
    float x_src = ((rx + 0.5) * scale) - 0.5;
    // valid neighbours [x_min.....x_max][ry]
    int x_min = std::max(0, (int)ceil(x_src) - window_sz);
    int x_max = std::min((int)floor(x_src) + window_sz, x - 1);

    for (int ry = 0; ry < res_y; ry++) {
      int px = rx * res_y + ry;
      float final_pixel = 0.0;
      for (int i = x_min; i <= x_max; i++) {
        // wt for i,ry
        // weight(x_src - x_min)
        float wt = precalc_l_x[(rx * (2 * window_sz + 1)) + i - x_min];
        // use scratch buffer [X * Scale*y]
        // TODO : this is bad for Cache-line..need to see how to improve
        final_pixel += (wt * (float)buf[i * res_y + ry]);
      }
      // trunc and clamp
      res[px] = (uint8_t)std::clamp((int)final_pixel, 0, 255);
    }
  }
  auto end_vert = std::chrono::steady_clock::now();

  double elapsed_ms_v =
      std::chrono::duration<double, std::milli>(end_vert - start_vert).count();

  LOG_INFO("vertical pass : " << elapsed_ms_v << " ms");

  std::free(buf);

  LOG_DEBUG("completed resize....");
  return res;
}