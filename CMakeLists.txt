cmake_minimum_required(VERSION 3.10.0)
project(lanczos VERSION 0.1.0 LANGUAGES C CXX)


add_executable(lanczos main.cpp
    external-libs/stb_image.h
    src/lanczos.cpp
    )

target_include_directories(lanczos PRIVATE ${PROJECT_SOURCE_DIR}/external-libs ${PROJECT_SOURCE_DIR}/include)
