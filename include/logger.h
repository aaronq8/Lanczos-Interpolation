#pragma once

#include <iostream>

#define LOG_INFO(msg)                                                          \
  do {                                                                         \
    std::cout << "[INFO] " << msg << " \n";                                    \
  } while (0)

#ifndef DISABLE_DEBUG_LOGS
#define LOG_DEBUG(msg)                                                         \
  do {                                                                         \
    std::cout << "[DEBUG] " << msg << " \n";                                   \
  } while (0)
#else
#define LOG_DEBUG(msg)                                                         \
  do {                                                                         \
  } while (0)
#endif