cmake -S . -B build/profile \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fno-omit-frame-pointer" \
    -DENABLE_ASAN=OFF

cmake --build build/profile -j