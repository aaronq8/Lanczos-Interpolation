#include "lanczos.h"
#include <iostream>

uint8_t *lanczos_resize(uint8_t *img, int x, int y) {
  std::cout << "----------------------\n";
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      std::cout << static_cast<int>(img[i * y + j]) << ",";
    }
    std::cout << "\n";
  }
  std::cout << "----------------------\n";
  return nullptr;
}