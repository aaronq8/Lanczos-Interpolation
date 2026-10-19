#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "lanczos.h"
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "Loading image from sample-images....\n";
  int x, y, org_c;
  constexpr int8_t c_num = 3;
  // [RGB][RGB][RGB]....
  stbi_uc *img = stbi_load("./sample-images/0.png", &x, &y, &org_c, 3);
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
      int pix_num = i * y + j;
      r_plane[pix_num] = img[c_num * pix_num];
      g_plane[pix_num] = img[c_num * pix_num + 1];
      b_plane[pix_num] = img[c_num * pix_num + 2];
    }
  }
  lanczos_resize(r_plane, x, y);
  lanczos_resize(g_plane, x, y);
  lanczos_resize(b_plane, x, y);
  stbi_image_free(img);
  return 0;
}
