#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "lanczos.h"
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "Loading image from sample-images....\n";
  int x, y, org_c;
  constexpr int8_t c_num = 3;
  // [RGB][RGB][RGB]....
  stbi_uc *img = stbi_load("./sample-images/0.jpg", &y, &x, &org_c, 3);
  // for now we only support 8-bit values
  assert(sizeof(img[0]) == 1);
  std::cout << "Loaded image....x : " << x << ", y : " << y
            << ", channels : " << org_c << "\n";
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

  uint8_t *res_r = lanczos_resize(r_plane, x, y);
  uint8_t *res_g = lanczos_resize(g_plane, x, y);
  uint8_t *res_b = lanczos_resize(b_plane, x, y);

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
  std::cout << "writing png...\n";
  if (stbi_write_png("sample-images/resized.png", res_y, res_x, c_num,
                     (const void *)res, c_num * res_y)) {
    std::cout << "generated resized.png.....\n";
  } else {
    std::cerr << "FAILED to generate resized.png...\n";
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
