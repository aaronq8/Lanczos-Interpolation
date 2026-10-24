#include "lanczos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

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

// stores [y_min,y_max_,y_src] for each col
int *pre_calc_y = nullptr;
// stores [x_min,x_max,x_src] for each row
int *pre_calc_x = nullptr;
/*
 * Scratch buffer to store intermediate values after horizontal pass
 * buf => (y*(1/scale)*x)
 */
float *buf = nullptr;

void lanczos_free() {
  std::free(pre_calc_x);
  std::free(pre_calc_y);
  std::free(buf);
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
  // we need to init 0, because need to load min(8,y_max-y_min+1) weights per
  // output px
  float *precalc_l_y =
      (float *)std::calloc(res_y * l_precalc_stride, sizeof(float));
  float *precalc_l_x =
      (float *)std::calloc(res_x * l_precalc_stride, sizeof(float));
  pre_calc_y = (int *)std::malloc(3 * res_y * sizeof(int));
  pre_calc_x = (int *)std::malloc(3 * res_x * sizeof(int));
  buf = (float *)std::calloc(res_y * (x + 2 * window_sz + 1), sizeof(float));

  for (int ry = 0; ry < res_y; ry++) {
    float y_src = ((ry + 0.5) * scale) - 0.5;
    // valid neighbours [rx][y_min....y_max]
    int y_min = std::max((int)ceil(y_src) - window_sz, 0);
    int y_max = std::min((int)floor(y_src) + window_sz, y - 1);
    int col_num = ry * l_precalc_stride;
    float tot_wt_y = 0.0f;
    //[y_min,y_max,y_src]...
    pre_calc_y[3 * ry] = y_min;
    pre_calc_y[3 * ry + 1] = y_max;
    pre_calc_y[3 * ry + 2] = y_src;
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
    pre_calc_x[3 * rx] = x_min;
    pre_calc_x[3 * rx + 1] = x_max;
    pre_calc_x[3 * rx + 2] = x_src;

    float tot_wt_x = 0.0f;
    // precalculate L3 values
    // x_min.....x_src.....x_max
    // ~50ms -> ~5ms => 900% faster
    int row_num = rx * l_precalc_stride;
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
void lanczos_resize(uint8_t *img, int x, int y, float *precalc_l_y,
                    float *precalc_l_x, uint8_t **res_ptr) {
  LOG_DEBUG("resizing....");
  // auto start_horz = std::chrono::steady_clock::now();

  int res_x = scale_rec * x;
  int res_y = scale_rec * y;
  uint8_t *res = *(res_ptr);
  /*
   * We will Linearly Seperate the Lanczos Kernel
   * Interpolate in y and store values in temp scratch buffer with row x scale *
   * width then Interpolate in x, this reduces number of L3 calls from ~36 -> ~6
   * per pixel (window_sz = 3 and interior pixels)
   * added padding to scratch buf for out-of-bounds handling during AVX vertical
   * pass
   */
  // horizontal pass
  // we go column major so that y_src is fixed
  // AVX-2
  /*
   * main idea is to use m256 registers to load full window per output px
   * [w0,w1,w2,w3,w4,w5,w6,0]
   * [p0,p1,p2,p3,p4,p5,p6,idk]
   * px = w0*p0+w1*p1+w2*p2+w3*p3+...+w6*p6
   */
  int rx = 0;
  for (; rx < x; rx++) {
    // valid neighbours [rx][y_min....y_max]
    for (int ry = 0; ry < res_y; ry++) {
      int y_min = pre_calc_y[3 * ry];
      int y_max = pre_calc_y[3 * ry + 1];
      float y_src = pre_calc_y[3 * ry + 2];
      int px = rx * res_y + ry;
      float buf_val = 0.0;
      //[w0,w1,w2,w3,w4,w5,w6,0]
      __m256 w = _mm256_loadu_ps(precalc_l_y + ry * l_precalc_stride);
      //[p0,p1....p7|x,x,x,x...x]
      __m128i p_low = _mm_loadl_epi64((__m128i_u *)(img + rx * y + y_min));
      // load 8x uint8 from lower 64 bits of p_low as 8xuint32
      __m256i p_int = _mm256_cvtepu8_epi32(p_low);
      //[p0,p1,p2,.....]
      __m256 p = _mm256_cvtepi32_ps(p_int);
      //[w0*p0,w1*p1,w2*p2,....,w6*p6,0*running_out_of_names_man]
      __m256 wxp = _mm256_mul_ps(w, p);
      // reduce
      __m128 wxp_low = _mm256_castps256_ps128(wxp);
      __m128 wxp_high = _mm256_extractf128_ps(wxp, 1);
      //[1+5,2+6,3+7,4+8]
      __m128 wxp_sum1 = _mm_add_ps(wxp_low, wxp_high);
      //[1+5+2+6,3+7+4+8|1+5+2+6,3+7+4+8]
      __m128 wxp_reduce = _mm_hadd_ps(wxp_sum1, wxp_sum1);
      //[1+..+8,1+..+8|1+..+8,1+..+8]
      __m128 wxp_full = _mm_hadd_ps(wxp_reduce, wxp_reduce);
      // wieghts were normalized during precalc
      buf[px] = _mm_cvtss_f32(wxp_full);
    }
  }

  // auto start_vert = std::chrono::steady_clock::now();
  // double elapsed_ms_h =
  //     std::chrono::duration<double, std::milli>(start_vert - start_horz)
  //         .count();

  // LOG_INFO("horizontal pass : " << elapsed_ms_h << " ms");

  // vertical pass
  // row-wise so that res access is cache aligned => ~4% faster
  // buf access still needs to be improved
  const __m256 zero = _mm256_setzero_ps();
  const __m256i zeroi = _mm256_setzero_si256();
  const __m256 mx_8 = _mm256_set1_ps(255.0f);

  for (int rx = 0; rx < res_x; rx++) {
    // valid neighbours [x_min.....x_max][ry]
    int x_min = pre_calc_x[rx * 3];
    int x_max = pre_calc_x[rx * 3 + 1];
    int ry = 0;
    // AVX-2
    /* Main idea is since weights are fixed for each neighbour in same row
     * w0*p0 w0*p1........w0*p7
     * w1*p0 w1*p1........w1*p7
     * w2*p0 w2*p1........w2*p7
     * ....
     * w6*p0 w6*p1........w6*p7
     */
    for (; ry + 7 <= res_y - 1; ry += 8) {
      //[0.0,0.0,...0.0]
      __m256 final_pix = _mm256_setzero_ps();
      // load p0...p7 at x_min+v...since window_Sz=3 fixed

#pragma unroll 7
      for (int v = 0; v <= x_max - x_min; v++) {
        // tried both...x_max-x_min faster by 3.79%
        __m256 p = _mm256_loadu_ps(buf + ((x_min + v) * res_y) + ry);
        // [wi,wi,....wi]
        __m256 wi = _mm256_set1_ps(precalc_l_x[(rx * l_precalc_stride) + v]);
        // add [wi*p0,wi*p1,wi*p2,...wi*p7] to final_pix
        final_pix = _mm256_fmadd_ps(p, wi, final_pix);
      }
      // convert float -> uint8
      __m256i final_pixi = _mm256_cvttps_epi32(final_pix);
      // 32-bit -> 16-bit
      // 0,1,2,3|4,5,6,7 => 0,1,2,3,0,0,0,0|4,5,6,7,0,0,0,0
      final_pixi = _mm256_packus_epi32(final_pixi, zeroi);
      // lower lane : 0,1,2,3,0,0.....0 (16 values,128-bit)
      // upper lane : 4,5,6,7,0,0,....0 (16 values,128-bit)
      // _us- usigned saturate will clamp [0-255]
      final_pixi = _mm256_packus_epi16(final_pixi, zeroi);
      __m128i low128 = _mm256_castsi256_si128(final_pixi);
      __m128i up128 = _mm256_extracti128_si256(final_pixi, 1);
      // [0,1,2,3] , [4,5,6,7]
      uint32_t low32 = _mm_cvtsi128_si32(low128);
      uint32_t up32 = _mm_cvtsi128_si32(up128);
      // little endian so actually stored as (3,2,1,0)
      // shift and extract 8 bits
      int px = rx * res_y + ry;
      res[px] = (uint8_t)(low32 & (0xFF));
      res[px + 1] = (uint8_t)((low32 >> 8) & (0xFF));
      res[px + 2] = (uint8_t)((low32 >> 16) & (0xFF));
      res[px + 3] = (uint8_t)((low32 >> 24) & 0xFF);
      // upper
      //(7,6,5,4)
      res[px + 4] = (uint8_t)((up32 & 0xFF));
      res[px + 5] = (uint8_t)((up32 >> 8) & (0xFF));
      res[px + 6] = (uint8_t)((up32 >> 16) & (0xFF));
      res[px + 7] = (uint8_t)((up32 >> 24) & 0xFF);
    }

    // scalar loop - this will run at most 7 times
    for (; ry < res_y; ry++) {
      int px = rx * res_y + ry;
      float final_pixel = 0.0;
      for (int i = x_min; i <= x_max; i++) {
        // wt for i,ry
        // weight(x_src - x_min)
        float wt = precalc_l_x[(rx * l_precalc_stride) + i - x_min];
        // use scratch buffer [X * Scale*y]
        // TODO : this is bad for Cache-line..need to see how to improve
        final_pixel += (wt * (float)buf[i * res_y + ry]);
      }
      // trunc and clamp
      res[px] = (uint8_t)std::clamp((int)final_pixel, 0, 255);
    }
  }
  // auto end_vert = std::chrono::steady_clock::now();

  // double elapsed_ms_v =
  //     std::chrono::duration<double, std::milli>(end_vert -
  //     start_vert).count();

  // LOG_INFO("vertical pass : " << elapsed_ms_v << " ms");

  LOG_DEBUG("completed resize....");
}