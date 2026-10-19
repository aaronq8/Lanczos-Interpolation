cmake_minimum_required(VERSION 3.10.0)
project(lanczos VERSION 0.1.0 LANGUAGES C CXX)

add_executable(lanczos main.cpp
    external-libs/stb_image.h
    src/lanczos.cpp
    external-libs/stb_image_write.h
    )

target_include_directories(lanczos PRIVATE ${PROJECT_SOURCE_DIR}/external-libs ${PROJECT_SOURCE_DIR}/include)

# disable logs on release build
target_compile_definitions(lanczos PRIVATE
    $<$<CONFIG:Release>:DISABLE_DEBUG_LOGS>
)

option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
# always disable ASAN on release builds
    target_compile_options(lanczos PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
    target_link_options(lanczos PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>)
endif()