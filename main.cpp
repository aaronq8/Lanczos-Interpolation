#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ratio>
#include <sys/types.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "lanczos.h"
#include "logger.h"

int main(int argc, char **argv) {
  LOG_INFO("Loading image from sample-images....");
  int x, y, org_c;
  constexpr int8_t c_num = 3;
  // [RGB][RGB][RGB]....
  stbi_uc *img = stbi_load("./sample-images/0.jpg", &y, &x, &org_c, 3);
  // for now we only support 8-bit values
  assert(sizeof(img[0]) == 1);
  LOG_INFO("Loaded image....x : " << x << ", y : " << y
                                  << ", channels : " << org_c);
  // seperate planes
  uint8_t *r_plane = (uint8_t *)std::malloc(x * y);
  uint8_t *g_plane = (uint8_t *)std::malloc(x * y);
  uint8_t *b_plane = (uint8_t *)std::malloc(x * y);

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      int px = i * y + j;
      r_plane[px] = img[c_num * px];
      g_plane[px] = img[c_num * px + 1];
      b_plane[px] = img[c_num * px + 2];
    }
  }

  auto start = std::chrono::steady_clock::now();
  uint8_t *res_r = lanczos_resize(r_plane, x, y);
  uint8_t *res_g = lanczos_resize(g_plane, x, y);
  uint8_t *res_b = lanczos_resize(b_plane, x, y);
  auto end = std::chrono::steady_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  LOG_INFO("Lanczos Kernel elapsed : " << elapsed_ms << " ms");

  int res_x = x * scale_rec, res_y = y * scale_rec;
  stbi_uc *res = (stbi_uc *)std::malloc(res_x * res_y * c_num);
  for (int i = 0; i < res_x; i++) {
    for (int j = 0; j < res_y; j++) {
      int px = i * res_y + j;
      //[RGB][RGB]....
      res[c_num * px] = res_r[px];
      res[c_num * px + 1] = res_g[px];
      res[c_num * px + 2] = res_b[px];
    }
  }
  LOG_INFO("writing png...\n");
  if (stbi_write_png("sample-images/resized.png", res_y, res_x, c_num,
                     (const void *)res, c_num * res_y)) {
    LOG_INFO("generated resized.png.....");
  } else {
    LOG_INFO("FAILED to generate resized.png...");
  }
  // free everything
  std::free(res);
  std::free(res_r);
  std::free(res_g);
  std::free(res_b);
  std::free(r_plane);
  std::free(g_plane);
  std::free(b_plane);
  stbi_image_free(img);

  return 0;
}
