cmake -S . -B build/profile \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fno-omit-frame-pointer" \
    -DENABLE_ASAN=OFF

cmake --build build/profile -j

./build/profile/lanczos-bench --benchmark_repetitions=10 --benchmark_min_time=0.5s --benchmark_report_aggregates_only=true