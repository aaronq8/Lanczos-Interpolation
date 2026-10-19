#include "lanczos.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

inline float sinc(float x) {
  // x==0
  if (std::abs(x) < 1e-9)
    return 1.0;
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
  std::cout << "resizing....\n";
  uint8_t *res = (uint8_t *)std::malloc(x * y * scale_rec * scale_rec);
  int res_x = scale_rec * x;
  int res_y = scale_rec * y;

  for (int rx = 0; rx < res_x; rx++) {
    for (int ry = 0; ry < res_y; ry++) {
      int px = rx * res_y + ry;
      float x_src = ((rx + 0.5) * scale) - 0.5;
      float y_src = ((ry + 0.5) * scale) - 0.5;
      // valid neighbours [x_min....x_max],[y_min....y_max]
      int x_min = std::max(0, (int)ceil(x_src) - window_sz);
      int x_max = std::min((int)floor(x_src) + window_sz, x - 1);
      int y_min = std::max((int)ceil(y_src) - window_sz, 0);
      int y_max = std::min((int)floor(y_src) + window_sz, y - 1);
      //   std::cout << "(" << x_min << " : " << x_max << "),(" << y_min << ":"
      //             << y_max << ")\n";
      float tot_wt = 0.0;
      float dest_val = 0.0;
      for (int i = x_min; i <= x_max; i++) {
        for (int j = y_min; j <= y_max; j++) {
          // calculate weight for neighbour at i,j
          float wt = L3(x_src - i) * L3(y_src - j);
          dest_val += (wt * (float)img[i * y + j]);
          tot_wt += wt;
        }
      }
      // normalize and clamp 0-255
      int norm =
          (std::abs(tot_wt) < 1e-9) ? (int)dest_val : (int)(dest_val / tot_wt);
      res[px] = (uint8_t)std::clamp(norm, 0, 255);
    }
  }
  std::cout << "completed resize....\n";
  return res;
}