cmake_minimum_required(VERSION 3.10.0)
project(lanczos VERSION 0.1.0 LANGUAGES C CXX)

# std::clamp only supported in clang c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(benchmark REQUIRED)

add_library(lanczos src/lanczos.cpp)

target_include_directories(lanczos PUBLIC ${PROJECT_SOURCE_DIR}/external-libs ${PROJECT_SOURCE_DIR}/include)

# disable logs on release build
target_compile_definitions(lanczos PUBLIC
    $<$<CONFIG:Release>:DISABLE_DEBUG_LOGS>
    $<$<CONFIG:RelWithDebInfo>:DISABLE_DEBUG_LOGS>
)

option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
# always disable ASAN on release builds
    target_compile_options(lanczos PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
    target_link_options(lanczos PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address>)
endif()

# generate executables
add_executable(lanczos-main main.cpp)
target_link_libraries(lanczos-main PRIVATE lanczos)

add_executable(lanczos-bench benchmarks/bench_lanczos.cpp)
target_link_libraries(lanczos-bench PRIVATE lanczos benchmark::benchmark_main)