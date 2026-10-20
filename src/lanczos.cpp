#include "lanczos.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "logger.h"

inline float sinc(float x) {
  // sinc(1) ~ -2*10^-8
  if (std::abs(x) < epsilon) return 1.0;
  return std::sin(PI * x) / (PI * x);
}

inline float L3(float x) {
  // Lanczos with window size=3
  return sinc(x) * sinc(x / (1.0f * window_sz));
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
uint8_t *lanczos_resize(uint8_t *img, int x, int y) {
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
  for (int rx = 0; rx < x; rx++) {
    for (int ry = 0; ry < res_y; ry++) {
      int px = rx * res_y + ry;
      float y_src = ((ry + 0.5) * scale) - 0.5;
      // valid neighbours [rx][y_min....y_max]
      int y_min = std::max((int)ceil(y_src) - window_sz, 0);
      int y_max = std::min((int)floor(y_src) + window_sz, y - 1);

      float tot_wt = 0.0;
      float buf_val = 0.0;
      for (int j = y_min; j <= y_max; j++) {
        // calculate weight for neighbour rx,j
        // L(0)=1 so we can ignore row/x weight
        float wt = L3(y_src - j);
        buf_val += (wt * (float)img[rx * y + j]);
        tot_wt += wt;
      }
      // normalize but store as float since we want to minimize precision loss
      // before 2nd pass
      buf[px] = (std::abs(tot_wt) < epsilon) ? buf_val : buf_val / tot_wt;
    }
  }

  // vertical pass
  for (int ry = 0; ry < res_y; ry++) {
    for (int rx = 0; rx < res_x; rx++) {
      int px = rx * res_y + ry;
      float x_src = ((rx + 0.5) * scale) - 0.5;
      // valid neighbours [x_min.....x_max][ry]
      int x_min = std::max(0, (int)ceil(x_src) - window_sz);
      int x_max = std::min((int)floor(x_src) + window_sz, x - 1);

      float tot_wt = 0.0;
      float final_pixel = 0.0;
      for (int i = x_min; i <= x_max; i++) {
        // wt for i,ry
        float wt = L3(x_src - i);
        // use scratch buffer [X * Scale*y]
        // TODO : this is bad for Cache-line..need to see how to improve
        final_pixel += (wt * (float)buf[i * res_y + ry]);
        tot_wt += wt;
      }
      int norm = (std::abs(tot_wt) < epsilon) ? (int)final_pixel
                                              : (int)(final_pixel / tot_wt);
      res[px] = (uint8_t)std::clamp(norm, 0, 255);
    }
  }

  std::free(buf);

  LOG_DEBUG("completed resize....");
  return res;
}